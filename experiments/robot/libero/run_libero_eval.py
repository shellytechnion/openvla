"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union
import yaml

import draccus
import numpy as np
import tqdm
import csv
sys.path.append('/mnt/pub/shellyf/tmp_openVLA')
sys.path.append('/mnt/pub/shellyf/tmp_openVLA/LIBERO')
#sys.path.append("/home/shellyf/Projects/openvla")
#sys.path.append("/home/shellyf/Projects/openvla/LIBERO")
# sys.path.append("LIBERO")
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (get_processor, add_text_to_image, probs_for_calibration, calculate_ece,
                                             calculate_ece_on_results)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    device: str = "cuda:0"                           # Device to run model on
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    #################################################################################################################
    # CALIBRATION - specific parameters
    #################################################################################################################
    # in each action there are 7 DOF, so the action calibration is done on all DOFs by applying action_calibration_type
    # and the episode calibration is done on all actions by applying episode_calibration_type
    action_calibration_type: str = "mul"      # how to calculate probabilities of each action. Options: mul, max, min, avg, perplexity,
    episode_calibration_type: str = "mul"     # how to calculate probabilities of each episode. Options: mul, max, min, avg, perplexity,
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    @staticmethod
    def from_yaml(file_path: Union[str, Path]) -> 'GenerateConfig':
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        # Filter out unexpected fields
        valid_fields = {f.name for f in fields(GenerateConfig)}
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return GenerateConfig(**filtered_config_dict)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"Calibration-{cfg.action_calibration_type}-{cfg.episode_calibration_type}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # add calibration dictionary results to csv
    csv_file_path = os.path.join(cfg.local_log_dir, run_id + ".csv")
    # Write calib_dict to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["episode_success", "episode_probs", "task"])

    print(f"Calibration results written to {csv_file_path}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            reinit=True
        )
        # run.define_metric("#steps", hidden=True)  # don't create a plot for "epoch"
        # run.define_metric("#episodes", hidden=True)  # don't create a plot for "epoch"

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")
        # Start episodes
        task_episodes, task_successes = 0, 0
        successes_ece = []
        episode_probs_ece = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            action_probs_arr = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action and probabilities
                    action, probs = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Save preprocessed image for replay video
                    img = add_text_to_image(img, f"Probs: {probs}")
                    replay_images.append(img)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # process the probabilities
                    action_prob = probs_for_calibration(probs, cfg.action_calibration_type)
                    action_probs_arr.append(action_prob)
                    if cfg.use_wandb and task_episodes % 3 == 0:
                        wandb.log(
                            {
                                f"Calibration_action/{task_description}_episode_{task_episodes+1}": float(action_prob),
                                "#steps": t,
                            }
                        )
                    log_file.write(f"# action No {t} Probability: {action_prob})\n")
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            # process total probability of the episode
            episode_prob = probs_for_calibration(action_probs_arr, cfg.episode_calibration_type)
            episode_probs_ece.append(episode_prob)
            successes_ece.append(float(done))
            ece = calculate_ece(episode_probs_ece, successes_ece)
            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"Episode probability: {episode_prob}")
            print(f"ECE: {ece}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episode probability: {episode_prob}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
            # Write results to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([done, episode_prob, task_description])

            if cfg.use_wandb:
                wandb.log(
                    {
                        f"Calibration_episode/{task_description}": float(episode_prob),
                        f"Calibration_episode/success_rate_{task_description}": float(done),
                        f"Evaluation/ECE_{task_description}": ece,
                        "#episodes": total_episodes,
                    }
                )

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                    "#episodes": total_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
        # Finish the run
        wandb.finish()

def evaluate_results():
    """
    Evaluate the results of the calibration: train a classifier and calculate ECE and acuracy
    """
    ### train a classifier
    import pandas as pd
    import os
    import glob
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Define the directory path
    directory_path = '/mnt/pub/shellyf/tmp_openVLA/experiments/logs'

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    random_seeds = [1243, 7884, 83, 921, 423, 684, 781, 9, 1, 13702]
    mul_list = {"ece": [], "accuracy": []}
    max_list = {"ece": [], "accuracy": []}
    min_list = {"ece": [], "accuracy": []}
    avg_list = {"ece": [], "accuracy": []}
    # Print the list of CSV files
    for csv_file in csv_files:
        for seed in random_seeds:
            # if "Calibration-avg-mul" not in csv_file:
            #     continue
            # Load the dataset
            dataset = pd.read_csv(csv_file)

            # Extract features and labels
            X = dataset[['episode_probs']].copy()  # Assuming 'episode_probs' is the feature
            y = dataset['episode_success'].astype(int)  # Assuming 'episode_success' is the label
            for i in range(X.shape[0]):
                var = X.iloc[i, 0]
                if var != 0:
                    X.iloc[i, 0] = abs(np.log(X.iloc[i, 0])) # log empirically works better
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X.values.tolist(), y.values.tolist(), test_size=0.2,
                                                                random_state=seed)

            # Train a logistic regression classifier
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)

            # Calculate the accuracy of the classifier
            accuracy = accuracy_score(y_test, y_pred)
            if y_pred.max() - y_pred.min() == 0:
                print(f"can't predict, Accuracy {csv_file.split(r'/')[-1]} : {accuracy:.2f}")
                accuracy = 0
                ece = 0
            else:
                save_name = csv_file.split(r'/')[-1].split(".")[0].replace("Calibration", "ECE") + f"_seed_{seed}.png"
                ece = calculate_ece_on_results(np.array(X_test).squeeze(axis=1), y_test, save_name=save_name)
                print(f"Accuracy {csv_file.split(r'/')[-1]} : {accuracy:.2f}, ECE: {ece:.2f}")
            if "mul-2025" in csv_file:
                mul_list["ece"].append(ece)
                mul_list["accuracy"].append(accuracy)
            elif "max-2025" in csv_file:
                max_list["ece"].append(ece)
                max_list["accuracy"].append(accuracy)
            elif "min-2025" in csv_file:
                min_list["ece"].append(ece)
                min_list["accuracy"].append(accuracy)
            elif "avg-2025" in csv_file:
                avg_list["ece"].append(ece)
                avg_list["accuracy"].append(accuracy)
    print(f"mul_list avg Accuracy: {np.average(mul_list['accuracy'])}, ECE: {np.average(mul_list['ece'])}")
    print(f"max_list avg Accuracy: {np.average(max_list['accuracy'])}, ECE: {np.average(max_list['ece'])}")
    print(f"min_list avg Accuracy: {np.average(min_list['accuracy'])}, ECE: {np.average(min_list['ece'])}")
    print(f"avg_list avg Accuracy: {np.average(avg_list['accuracy'])}, ECE: {np.average(avg_list['ece'])}")


if __name__ == "__main__":

    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # try:
    #     config = GenerateConfig.from_yaml('LIBERO/libero/configs/config.yaml')
    # except FileNotFoundError:
    #     config = GenerateConfig.from_yaml("/home/shellyf/Projects/openvla/LIBERO/libero/configs/config.yaml")
    # action_calibration_types = ["mul", "max", "min", "avg"]
    # episode_calibration_types = ["mul", "max", "min", "avg"]
    # for action_type in action_calibration_types:
    #     for episode_type in episode_calibration_types:
    #         config.action_calibration_type = action_type
    #         config.episode_calibration_type = episode_type
    #         eval_libero(config)

    evaluate_results()



## from https://github.com/google-research/google-research/blob/master/language_model_uncertainty/KnowNo_TabletopSim.ipynb
# load from the csv the data
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # # load the data
#     dataset = pd.read_csv('/mnt/pub/shellyf/tmp_openVLA/experiments/logs/Calibration-mul-mul-2025_01_02-10_59_53.csv')
#
#     calculate_ece(dataset['episode_success'], dataset['episode_probs'], num_bins=10)
# #@markdown Then, get the non-conformity scores from the calibration set, which is 1 minus the likelihood of the **true** option,
# def temperature_scaling(logits, temperature):
#     logits = np.array(logits)
#     logits /= temperature
#
#     # apply softmax
#     logits -= logits.max()
#     logits = logits - np.log(np.sum(np.exp(logits)))
#     smx = np.exp(logits)
#     return smx
#
# non_conformity_score = []
# for data in dataset:
#   top_logprobs = data['top_logprobs']
#   top_tokens = data['top_tokens']
#   true_options = data['true_options']
#
#   # normalize the five scores to sum of 1
#   mc_smx_all = temperature_scaling(top_logprobs, temperature=5)
#
#   # get the softmax value of true option
#   true_label_smx = [mc_smx_all[token_ind]
#                     for token_ind, token in enumerate(top_tokens)
#                     if token in true_options]
#   true_label_smx = np.max(true_label_smx)
#
#   # get non-comformity score
#   non_conformity_score.append(1 - true_label_smx)
#   # find the quantile value qhat
#   q_level = np.ceil((num_calibration + 1) * (1 - epsilon)) / num_calibration
#   qhat = np.quantile(non_conformity_score, q_level, method='higher')
#   print('Quantile value qhat:', qhat)
#   print('')
#
#   # plot histogram and quantile
#   plt.figure(figsize=(6, 2))
#   plt.hist(non_conformity_score, bins=30, edgecolor='k', linewidth=1)
#   plt.axvline(
#       x=qhat, linestyle='--', color='r', label='Quantile value'
#   )
#   plt.title(
#       'Histogram of non-comformity scores in the calibration set'
#   )
#   plt.xlabel('Non-comformity score')
#   plt.legend();
#   plt.show()
#   print('')
#   print('A good predictor should have low non-comformity scores, concentrated at the left side of the figure'