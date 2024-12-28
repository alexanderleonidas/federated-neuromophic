import json
import pandas as pd
import numpy as np

directory = "results/"
file_name = "dp_epsilon0point1_training_scores"

# Load the JSON data
with open(directory + file_name + ".json", "r") as file:
    data = json.load(file)

# Initialize lists to store metrics for each epoch across all runs
num_epochs = 10
training_accuracy_all = [[] for _ in range(num_epochs)]
validation_accuracy_all = [[] for _ in range(num_epochs)]
training_precision_all = [[] for _ in range(num_epochs)]
validation_precision_all = [[] for _ in range(num_epochs)]
training_recall_all = [[] for _ in range(num_epochs)]
validation_recall_all = [[] for _ in range(num_epochs)]
training_f1_all = [[] for _ in range(num_epochs)]
validation_f1_all = [[] for _ in range(num_epochs)]

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
    training_accuracy = run_data["training_scores"]["Training Accuracy"]
    validation_accuracy = run_data["training_scores"]["Validation Accuracy"]
    training_precision = [epoch["precision"] for epoch in run_data["training_scores"]["Training Precision"]]
    validation_precision = [epoch["precision"] for epoch in run_data["training_scores"]["Validation Precision"]]
    training_recall = [epoch["recall"] for epoch in run_data["training_scores"]["Training Recall"]]
    validation_recall = [epoch["recall"] for epoch in run_data["training_scores"]["Validation Recall"]]
    training_f1 = [epoch["f1"] for epoch in run_data["training_scores"]["Training F1"]]
    validation_f1 = [epoch["f1"] for epoch in run_data["training_scores"]["Validation F1"]]

    for epoch in range(num_epochs):
        training_accuracy_all[epoch].append(training_accuracy[epoch])
        validation_accuracy_all[epoch].append(validation_accuracy[epoch])
        training_precision_all[epoch].append(training_precision[epoch])
        validation_precision_all[epoch].append(validation_precision[epoch])
        training_recall_all[epoch].append(training_recall[epoch])
        validation_recall_all[epoch].append(validation_recall[epoch])
        training_f1_all[epoch].append(training_f1[epoch])
        validation_f1_all[epoch].append(validation_f1[epoch])

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
        "Mean Training Accuracy": np.mean(training_accuracy_all[epoch]),
        "Std Training Accuracy": np.std(training_accuracy_all[epoch]),
        "Mean Validation Accuracy": np.mean(validation_accuracy_all[epoch]),
        "Std Validation Accuracy": np.std(validation_accuracy_all[epoch]),
        "Mean Training Precision": np.mean(training_precision_all[epoch]),
        "Std Training Precision": np.std(training_precision_all[epoch]),
        "Mean Validation Precision": np.mean(validation_precision_all[epoch]),
        "Std Validation Precision": np.std(validation_precision_all[epoch]),
        "Mean Training Recall": np.mean(training_recall_all[epoch]),
        "Std Training Recall": np.std(training_recall_all[epoch]),
        "Mean Validation Recall": np.mean(validation_recall_all[epoch]),
        "Std Validation Recall": np.std(validation_recall_all[epoch]),
        "Mean Training F1": np.mean(training_f1_all[epoch]),
        "Std Training F1": np.std(training_f1_all[epoch]),
        "Mean Validation F1": np.mean(validation_f1_all[epoch]),
        "Std Validation F1": np.std(validation_f1_all[epoch]),
    })
df_epoch = pd.DataFrame(epoch_results)
df_epoch.to_csv(directory + file_name + "epoch_metrics_summary.csv", index=False)

print("Overall and per-epoch metrics summaries saved to 'overall_metrics_summary.csv' and 'epoch_metrics_summary.csv'.")