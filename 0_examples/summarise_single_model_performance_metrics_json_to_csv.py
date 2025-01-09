import json

import numpy as np
import pandas as pd

directory = "../results/"
file_name = "dp_epsilon0point1_training_scores"

# Load the JSON data
with open(directory + file_name + ".json", "r") as file:
    data = json.load(file)

# Initialize lists to store metrics for each epoch across all runs
num_epochs = 10
training_loss_all = [[] for _ in range(num_epochs)]
validation_loss_all = [[] for _ in range(num_epochs)]
training_accuracy_all = [[] for _ in range(num_epochs)]
validation_accuracy_all = [[] for _ in range(num_epochs)]

# Initialize lists for overall metrics
overall_accuracy_all = []
overall_precision_all = []
overall_recall_all = []
overall_f1_all = []

# Iterate through each run in the JSON
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

    # Extract per-epoch metrics
    for run_id, run_data in data.items():
        training_loss = run_data["training_scores"]["Training Loss"]
        validation_loss = run_data["training_scores"]["Validation Loss"]
        training_accuracy = run_data["training_scores"]["Training Accuracy"]
        validation_accuracy = run_data["training_scores"]["Validation Accuracy"]

        # Append values for each epoch
        for epoch in range(num_epochs):
            training_loss_all[epoch].append(training_loss[epoch])
            validation_loss_all[epoch].append(validation_loss[epoch])
            training_accuracy_all[epoch].append(training_accuracy[epoch])
            validation_accuracy_all[epoch].append(validation_accuracy[epoch])

# Compute overall metrics (mean and std across all runs)
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
df_overall.to_csv(directory + file_name + "overall_metrics_summary.csv", index=False)

# Compute per-epoch metrics (mean and std for each epoch across all runs)
epoch_results = []
for epoch in range(num_epochs):
    epoch_results.append({
        "Epoch": epoch + 1,
        "Mean Training Loss": np.mean(training_loss_all[epoch]),
        "Std Training Loss": np.std(training_loss_all[epoch]),
        "Mean Validation Loss": np.mean(validation_loss_all[epoch]),
        "Std Validation Loss": np.std(validation_loss_all[epoch]),
        "Mean Training Accuracy": np.mean(training_accuracy_all[epoch]),
        "Std Training Accuracy": np.std(training_accuracy_all[epoch]),
        "Mean Validation Accuracy": np.mean(validation_accuracy_all[epoch]),
        "Std Validation Accuracy": np.std(validation_accuracy_all[epoch]),
    })
df_epoch = pd.DataFrame(epoch_results)
df_epoch.to_csv(directory + file_name + "epoch_metrics_summary.csv", index=False)

print("Overall and per-epoch metrics summaries saved to 'overall_metrics_summary.csv' and 'epoch_metrics_summary.csv'.")