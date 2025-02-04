import json
import pandas as pd
import numpy as np

# Load the JSON data from a file
with open('../results/training_scores_federated_fa_50runs.json', 'r') as f:
    data = json.load(f)

# Prepare a dictionary to store aggregated metrics
aggregated_metrics = {}

# Iterate over each experiment
for experiment_id, experiment_data in data.items():
    training_scores = experiment_data.get("training_scores", {})

    # Iterate over each epoch in the experiment
    for epoch, clients in training_scores.items():
        if epoch not in aggregated_metrics:
            aggregated_metrics[epoch] = {'Training Loss': [], 'Training Accuracy': [],
                                         'Validation Loss': [], 'Validation Accuracy': []}

        # Collect all client metrics for the epoch
        for client_id, metrics in clients.items():
            aggregated_metrics[epoch]['Training Loss'].extend(metrics['Training Loss'])
            aggregated_metrics[epoch]['Training Accuracy'].extend(metrics['Training Accuracy'])
            aggregated_metrics[epoch]['Validation Loss'].extend(metrics['Validation Loss'])
            aggregated_metrics[epoch]['Validation Accuracy'].extend(metrics['Validation Accuracy'])

# Calculate averages and standard deviations
results = []
for epoch, metrics in aggregated_metrics.items():
    avg_training_loss = np.mean(metrics['Training Loss'])
    std_training_loss = np.std(metrics['Training Loss'])

    avg_training_accuracy = np.mean(metrics['Training Accuracy'])
    std_training_accuracy = np.std(metrics['Training Accuracy'])

    avg_validation_loss = np.mean(metrics['Validation Loss'])
    std_validation_loss = np.std(metrics['Validation Loss'])

    avg_validation_accuracy = np.mean(metrics['Validation Accuracy'])
    std_validation_accuracy = np.std(metrics['Validation Accuracy'])

    results.append({
        'Epoch': epoch,
        'Avg Training Loss': avg_training_loss,
        'Std Training Loss': std_training_loss,
        'Avg Training Accuracy': avg_training_accuracy,
        'Std Training Accuracy': std_training_accuracy,
        'Avg Validation Loss': avg_validation_loss,
        'Std Validation Loss': std_validation_loss,
        'Avg Validation Accuracy': avg_validation_accuracy,
        'Std Validation Accuracy': std_validation_accuracy
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv('../results/aggregated_metrics.csv', index=False)