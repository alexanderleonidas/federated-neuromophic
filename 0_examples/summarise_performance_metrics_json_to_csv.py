import json
import pandas as pd
import numpy as np

# Load the JSON data
with open("data.json", "r") as file:
    data = json.load(file)

# Initialize lists to store metrics for each epoch across all runs
num_epochs = 10
training_loss_all = [[] for _ in range(num_epochs)]
validation_loss_all = [[] for _ in range(num_epochs)]
training_accuracy_all = [[] for _ in range(num_epochs)]
validation_accuracy_all = [[] for _ in range(num_epochs)]

# Iterate through each run in the JSON
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

# Calculate mean and std for each epoch across all runs
results = []
for epoch in range(num_epochs):
    results.append({
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

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_file = "epoch_metrics_summary.csv"
df.to_csv(output_file, index=False)

print(f"Epoch metrics summary saved to {output_file}")