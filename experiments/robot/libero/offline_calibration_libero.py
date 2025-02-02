import pandas as pd
from timm.models.maxxvit import maxvit_tiny_pm_256
import numpy as np
from experiments.robot.openvla_utils import (add_text_to_image, probs_for_calibration, calculate_ece)
from experiments.robot.calibration_utils import calculate_ece_on_results, calc_conformal_prediction

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to get all probabilities for a specific task
def get_probs_by_task(task_name):
    task_data = data[data['task'] == task_name]
    probs = task_data[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6']]
    success = task_data['Success'][0]
    return probs, success

# Function to get all probabilities for a specific episode
def get_probs_by_episode(episode_index):
    episode_data = data[data['episode_index'] == episode_index]
    probs = episode_data[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6']]
    return probs

def evaluate_results_offline(dataset, episode_calibration, action_calibration):
    random_seeds = [1243, 7884, 83, 921, 423, 684, 781, 9, 1, 13702]
    # calculate action and episode probabilities
    X = [] # episode probs
    Y = [] # episode success
    for task in dataset['task'].unique():
        task_data = dataset[dataset['task'] == task]
        for episode in task_data['episode_index'].unique():
            episode_data = task_data[task_data['episode_index'] == episode]
            action_probs = episode_data.apply(lambda row: probs_for_calibration(row[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6']], action_calibration), axis=1)
            episode_success= episode_data['Success'].iloc[0]
            episode_prob = probs_for_calibration(action_probs.values.tolist(), episode_calibration)
            X.append(episode_prob)
            Y.append(episode_success)

    X_new = pd.DataFrame(X)
    X_new = X_new.applymap(lambda var: abs(np.log(var)) if var != 0 else 0)
    avg_accuracy = 0
    avg_ece = 0
    for seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(X_new.values.tolist(), Y, test_size=0.2,random_state=seed)
#
        # Train a logistic regression classifier
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        if y_pred.max() - y_pred.min() == 0:
            print(f"can't predict, Accuracy to episode_calibration = {episode_calibration}, action_calibration = {action_calibration} : {accuracy:.2f}")
            accuracy = 0
            ece = 0
        else:
            save_name = f"ECE_libero_10_episode_{episode_calibration}_action_{action_calibration}_seed_{seed}.png"
            ece = calculate_ece_on_results(np.array(X_test).squeeze(axis=1), y_test, save_name=save_name)
            print(f"Accuracy to episode_calibration = {episode_calibration}, action_calibration = {action_calibration}  : {accuracy:.2f}, ECE: {ece:.2f}")

        avg_accuracy += accuracy
        avg_ece += ece
    avg_accuracy /= len(random_seeds)
    avg_ece /= len(random_seeds)

    calc_conformal_prediction(None, save=False, offline_confidences=X_new.values.tolist(), offline_successes=Y)
    print(" #############################################")
    print(
        f"episode_calibration = {episode_calibration}, action_calibration = {action_calibration} avg Accuracy: {avg_accuracy}, ECE: {avg_ece}")
    print(" #############################################")

# if __name__ == '__main__':
#     # Read the CSV file
#     csv_path = '/home/shellyf/Documents/research_results/libero_10_all_run_probs.csv'
#     data = pd.read_csv(csv_path)
#     # Calculate the minimum time step so there will be no time dependency
#     max_time_step = []
#     for task in data['task'].unique():
#         task_data = data[data['task'] == task]
#         for episode_index in task_data['episode_index'].unique():
#             episode_data = task_data[task_data['episode_index'] == episode_index]
#             max_time_step.append(episode_data['time_step'].max())
#     min_time_step = min(max_time_step)
#
#     # for task in data['task'].unique():
#         # task_data = data[data['task'] == task]
#     probabilities = data.loc[data['time_step'] <= min_time_step]
#     action_calibration_types = ["min", "avg", "mul", "max"]
#     episode_calibration_types = ["mul", "avg", "max", "min"]
#     for action_type in action_calibration_types:
#         for episode_type in episode_calibration_types:
#             evaluate_results_offline(probabilities, episode_type, action_type)

