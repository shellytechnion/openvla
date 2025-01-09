import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalize_confidences(confidences):
    """
    Normalize the confidences to be between 0 and 1.
    because the log probability is reversed (small prob -> high log and vice versa), we swap the order of the confidences
    """
    if len(np.where(confidences == 0)) > 0:
        max_conf = max(confidences) + (max(confidences) - min(confidences)) / 10.0
        confidences = np.array([x if x != 0 else max_conf for x in confidences])
    # normalize the confidences to be between 0 and 1

    confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())
    # swap the order because of the log
    return 1 - confidences

def calculate_ece_on_results(confidences, success_rate, num_bins=10, save_name=None):
    """
    Computes the Expected Calibration Error (ECE).
    Args:
        confidences (list): List of confidence scores corresponding to predictions (0 to 1).
        success_rate (list): List of ground truth labels.
        num_bins (int): Number of bins for bucketing predictions by confidence.
    Returns:
        float: Expected Calibration Error (ECE).
    """

    if len(success_rate) != len(confidences) :
        raise ValueError("Length of confidences, and ground_truth must be the same.")
    # normalize the confidences to be between 0 and 1
    confidences = normalize_confidences(confidences)
    # Initialize bins
    bin_boundaries = np.linspace(min(confidences), max(confidences), num_bins)
    bin_indices = np.digitize(confidences, bin_boundaries, right=False) - 1 # Adjust index to 0-based
    ece = 0.0
    empty_bins = 0
    # save bin_avg_success_rate and bin_avg_confidence for plotting
    bin_avg_success_rates = []
    bin_avg_confidences = []
    # Iterate through each bin
    for bin_idx in range(num_bins):
        # Get indices of predictions in the current bin
        bin_mask = bin_indices == bin_idx
        if np.sum(bin_mask) == 0:
            empty_bins += 1
            continue  # Skip empty bins
        # Get accuracy and average confidence for the bin
        bin_confidences = np.array(confidences)[bin_mask]
        bin_success_rate = np.array(success_rate)[bin_mask]
        bin_avg_confidence = np.mean(bin_confidences)
        bin_avg_success_rate = np.mean(bin_success_rate)

        bin_avg_success_rates.append(bin_avg_success_rate)
        bin_avg_confidences.append(bin_avg_confidence)
        # Compute contribution to ECE
        ece += np.abs(bin_avg_confidence - bin_avg_success_rate)
    try:
        returned_ece = ece / (num_bins - empty_bins)
    except ZeroDivisionError:
        returned_ece = 1.0
    # Plot the graph
    if save_name is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(bin_avg_confidences, bin_avg_success_rates, marker='o', linestyle='-')
        plt.plot([0, 1], [0, 1], color='pink', linestyle='-', linewidth=2)  # Add y = x line in pink
        plt.xlabel('Bin Average Confidence')
        plt.ylabel('Bin Average Success Rate')
        plt.title('Bin Average Confidence vs Bin Average Success Rate, ECE = {:.4f}'.format(returned_ece))
        plt.grid(True)
        plt.savefig(save_name)
        plt.show()

    return returned_ece

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

def calc_conformal_prediction(csv_path):
    ## from https://github.com/google-research/google-research/blob/master/language_model_uncertainty/KnowNo_TabletopSim.ipynb
    dataset = pd.read_csv(csv_path)
    # manipulate confidences to be between 0 and 1
    confidences = np.array(dataset[['episode_probs']].copy().values.tolist()).squeeze(axis=1)
    for i, var in enumerate(confidences):
        if var != 0:
            confidences[i] = abs(np.log(var))  # log empirically works better


    # normalize the confidences to be between 0 and 1
    confidences = normalize_confidences(confidences)

    true_confidences_indices = np.where(np.array(dataset['episode_success'].astype(int)) == 1)
    true_confidences = confidences[true_confidences_indices]
    false_confidences = confidences[np.where(np.array(dataset['episode_success'].astype(int)) == 0)]
    qhat = np.quantile(false_confidences, 0.95, method='lower')
    print(f'Quantile value qhat for {csv_path.split("Calibration-")[-1]}: {qhat}')

    # plot histogram and quantile
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(false_confidences, bins=30, edgecolor='k', linewidth=1, align="left")
    plt.axvline(
          x=qhat, linestyle='--', color='r', label='Quantile value'
      )
    plt.title(
          f'q-value of the false class in {csv_path.split("Calibration-")[-1]} '
      )
    plt.xlabel('confidence score')
    plt.ylabel('frequency')
    plt.legend()
    # plt.savefig(f'q-value of the false class in {csv_path.split("Calibration-")[-1].split(".")[0]}.png')

    plt.figure()
    plt.hist(true_confidences, bins=30, edgecolor='k', linewidth=1, align="left")
    plt.axvline(
          x=qhat, linestyle='--', color='r', label='Quantile value'
      )
    plt.title(
          f'q-value of the true class in {csv_path.split("Calibration-")[-1]}'
      )
    plt.xlabel('confidence score')
    plt.ylabel('frequency')
    plt.legend()
    # plt.savefig(f'q-value of the true class in {csv_path.split("Calibration-")[-1].split(".")[0]}.png')
    plt.show()
