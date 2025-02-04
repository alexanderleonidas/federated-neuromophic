import json
import numpy as np
import pandas as pd


def summarise_central_experiment(experiment):
    # Directory and file name
    directory = "../results/"
    file_name = ""

    # Load the JSON data
    with open(directory + file_name + ".json", "r") as file:
        data = json.load(file)

    # Initialize lists for overall metrics
    overall_accuracy_all = []
    overall_precision_all = []
    overall_recall_all = []
    overall_f1_all = []

    # Initialize dictionary for per-epoch metrics
    epoch_metrics = {}

    # Process each experiment
    for run_id, run_data in data.items():
        # Extract overall metrics
        total = run_data["final_metrics"]["total"]
        correct = run_data["final_metrics"]["correct"]
        overall_accuracy = correct / total
        overall_precision = run_data["final_metrics"]["precision"]
        overall_recall = run_data["final_metrics"]["recall"]
        overall_f1_score = run_data["final_metrics"]["f1_score"]

        overall_accuracy_all.append(overall_accuracy)
        overall_precision_all.append(overall_precision)
        overall_recall_all.append(overall_recall)
        overall_f1_all.append(overall_f1_score)

        # Extract training and validation metrics
        training_loss = run_data["training_scores"]["Training Loss"]
        validation_loss = run_data["training_scores"]["Validation Loss"]
        training_accuracy = run_data["training_scores"]["Training Accuracy"]
        validation_accuracy = run_data["training_scores"]["Validation Accuracy"]

        num_epochs = len(training_loss)  # Dynamically determine number of epochs

        # Ensure epoch_metrics has lists initialized for all epochs
        for epoch in range(num_epochs):
            if epoch not in epoch_metrics:
                epoch_metrics[epoch] = {
                    "Training Loss": [],
                    "Validation Loss": [],
                    "Training Accuracy": [],
                    "Validation Accuracy": [],
                }

            # Append metrics for this epoch
            epoch_metrics[epoch]["Training Loss"].append(training_loss[epoch])
            epoch_metrics[epoch]["Validation Loss"].append(validation_loss[epoch])
            epoch_metrics[epoch]["Training Accuracy"].append(training_accuracy[epoch])
            epoch_metrics[epoch]["Validation Accuracy"].append(validation_accuracy[epoch])

    # Compute overall metrics (mean and std across all experiments)
    overall_results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Mean": [
            np.mean(overall_accuracy_all),
            np.mean(overall_precision_all),
            np.mean(overall_recall_all),
            np.mean(overall_f1_all),
        ],
        "Standard Deviation": [
            np.std(overall_accuracy_all),
            np.std(overall_precision_all),
            np.std(overall_recall_all),
            np.std(overall_f1_all),
        ],
    }
    df_overall = pd.DataFrame(overall_results)
    df_overall.to_csv(directory + file_name + "_overall_metrics_summary.csv", index=False)

    # Compute per-epoch metrics (mean and std for each epoch across all experiments)
    epoch_results = []
    for epoch, metrics in epoch_metrics.items():
        epoch_results.append({
            "Epoch": epoch + 1,
            "Mean Training Loss": np.mean(metrics["Training Loss"]),
            "Std Training Loss": np.std(metrics["Training Loss"]),
            "Mean Validation Loss": np.mean(metrics["Validation Loss"]),
            "Std Validation Loss": np.std(metrics["Validation Loss"]),
            "Mean Training Accuracy": np.mean(metrics["Training Accuracy"]),
            "Std Training Accuracy": np.std(metrics["Training Accuracy"]),
            "Mean Validation Accuracy": np.mean(metrics["Validation Accuracy"]),
            "Std Validation Accuracy": np.std(metrics["Validation Accuracy"]),
        })
    df_epoch = pd.DataFrame(epoch_results)
    df_epoch.to_csv(directory + file_name + "_epoch_metrics_summary.csv", index=False)

    print(
        "Overall and per-epoch metrics summaries saved to '_overall_metrics_summary.csv' and '_epoch_metrics_summary.csv'.")