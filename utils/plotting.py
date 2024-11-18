import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from sklearn.metrics import roc_curve

from data.experiment_data.JSON.json_manager import JSONManager

from training.watchers.federated_training_watcher import FederatedTrainingWatcher
from utils.globals import MAX_EPOCHS, NUM_ROUNDS, NUM_CLIENTS, PATH_TO_SAVED_RESULTS

dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_learning_curve(training_scores, num_epochs=MAX_EPOCHS):
    x = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    # Plot loss curves
    plot_loss_curve(ax1, x, training_scores)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss / Epoch'); plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    # Plot accuracy curves
    plot_accuracy_curve(ax2, x, training_scores)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy / Epoch'); plt.legend()

    plt.tight_layout()
    plt.show()


def plot_loss_curve(ax, x, training_scores):
    ax.plot(x, training_scores['Training Loss'], label='Training Loss', color=dfc[0])
    ax.plot(x, training_scores['Validation Loss'], label='Validation Loss', color=dfc[1])


def plot_accuracy_curve(ax, x, training_scores):
    ax.plot(x, training_scores['Training Accuracy'], label='Training Accuracy', color=dfc[0])
    ax.plot(x, training_scores['Validation Accuracy'], label='Validation Accuracy', color=dfc[1])


def plot_clients_learning_curves(client_round_watcher: FederatedTrainingWatcher,num_epochs=MAX_EPOCHS*NUM_ROUNDS):

    stacked_client_scores = stack_client_scores(client_round_watcher.round_records)

    x = range(1, num_epochs + 1)

    if NUM_CLIENTS <= 1:
        return plot_learning_curve(stacked_client_scores[0], num_epochs=num_epochs)

    fig, axs = plt.subplots(figsize=(5*NUM_CLIENTS, 8), nrows=2, ncols=NUM_CLIENTS, sharex='col', sharey='row')

    for i, (client_id, client_score) in enumerate(stacked_client_scores.items()):
        # Plot loss curve on the first row
        ax_loss = axs[0, i]
        plot_loss_curve(ax_loss, x, client_score)

        # Plot accuracy curve on the second row
        ax_acc = axs[1, i]
        plot_accuracy_curve(ax_acc, x, client_score)

        # Add vertical dashed lines at positions where i % MAX_EPOCHS == 0
        for epoch in range(MAX_EPOCHS, num_epochs + 1, MAX_EPOCHS):
            ax_loss.axvline(x=epoch, linestyle='--', color='gray')
            ax_acc.axvline(x=epoch, linestyle='--', color='gray')

    # Set individual titles for 'Loss' and 'Accuracy' based on the row
    for row, label in zip([0, 1], ['Loss / Epoch', 'Accuracy / Epoch']):
        for ax in axs[row, :]:
            ax.set_title(label, loc='left', fontsize='medium')

    # Remove individual subplot titles to avoid overlap
    for ax in axs.flat:
        ax.title.set_position([0.5, 1.0])

    # Set common xlabel for the entire figure
    fig.supxlabel('Epochs', fontsize='large')

    # Set common ylabels for each row
    axs[0, 0].set_ylabel('Loss', fontsize='large')
    axs[1, 0].set_ylabel('Accuracy %', fontsize='large')

    plt.subplots_adjust(top=0.80, hspace=0.3, left=0.05, right=0.85)

    # Add a general title for the entire figure
    fig.suptitle('Clients Learning Curves', fontsize=16, y=0.97)


    # Add common column titles 'Client {i}'
    for i in range(NUM_CLIENTS):
        pos = axs[0, i].get_position()
        x_pos = pos.x0 + pos.width / 2
        fig.text(x_pos, 0.85, f'Client {i + 1}', ha='center', va='bottom', fontsize='large')

    legend_labels = ['Training', 'Validation', 'Round Averaging']
    handles = [plt.Line2D([0], [0], color=dfc[0]), plt.Line2D([0], [0], color=dfc[1]), plt.Line2D([0], [0], color='gray', linestyle='--')]
    fig.legend(handles, legend_labels, loc='upper right', fontsize='large', bbox_to_anchor=(1.00, 0.5))

    plt.show()


def stack_client_scores(round_records):
    client_scores = {}

    for client_id in range(NUM_CLIENTS):
        client_scores[client_id] = {
            'Training Loss': [],
            'Validation Loss': [],
            'Training Accuracy': [],
            'Validation Accuracy': []
        }

    for round_num in range(NUM_ROUNDS):
        for client_id in range(NUM_CLIENTS):
            scores = round_records[round_num][client_id]
            for key in ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']:
                client_scores[client_id][key].extend(scores[key])


    return client_scores


def plot_server_round_scores(server_round_scores):
    pass

# Function to calculate mean metrics
def calculate_mean_metrics(results):
    total_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1_score": [],
        "class_wise_metrics": {str(i): {"precision": [], "recall": [], "f1-score": []} for i in range(10)}
    }

    for experiment_set in results:
        if isinstance(experiment_set, list):
            for experiment in experiment_set:
                total_metrics["accuracy"].append(experiment["correct"] / experiment["total"])
                total_metrics["precision"].append(experiment["precision"])
                total_metrics["recall"].append(experiment["recall"])
                total_metrics["f1_score"].append(experiment["f1_score"])

                for class_id, metrics in experiment["class_wise_metrics"].items():
                    if class_id.isdigit():
                        total_metrics["class_wise_metrics"][class_id]["precision"].append(metrics["precision"])
                        total_metrics["class_wise_metrics"][class_id]["recall"].append(metrics["recall"])
                        total_metrics["class_wise_metrics"][class_id]["f1-score"].append(metrics["f1-score"])
        else:
            total_metrics["accuracy"].append(experiment_set["correct"] / experiment_set["total"])
            total_metrics["precision"].append(experiment_set["precision"])
            total_metrics["recall"].append(experiment_set["recall"])
            total_metrics["f1_score"].append(experiment_set["f1_score"])

            for class_id, metrics in experiment_set["class_wise_metrics"].items():
                if class_id.isdigit():
                    total_metrics["class_wise_metrics"][class_id]["precision"].append(metrics["precision"])
                    total_metrics["class_wise_metrics"][class_id]["recall"].append(metrics["recall"])
                    total_metrics["class_wise_metrics"][class_id]["f1-score"].append(metrics["f1-score"])

    mean_metrics = {
        "accuracy": np.mean(total_metrics["accuracy"]),
        "precision": np.mean(total_metrics["precision"]),
        "recall": np.mean(total_metrics["recall"]),
        "f1_score": np.mean(total_metrics["f1_score"]),
        "class_wise_metrics": {class_id: {
            "precision": np.mean(metrics["precision"]),
            "recall": np.mean(metrics["recall"]),
            "f1-score": np.mean(metrics["f1-score"])
        } for class_id, metrics in total_metrics["class_wise_metrics"].items()}
    }
    return mean_metrics, total_metrics

def compute_tpr_fpr(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return tpr, fpr

def plot_metrics():
    jsonManager = JSONManager()
    result_arrays = jsonManager.load_json_files_from_directory(PATH_TO_SAVED_RESULTS)
    print(f"Loaded {len(result_arrays)} result sets.")
    print("result_arrays", result_arrays)
    mean_metrics_list = []
    total_metrics_list = []
    for result in result_arrays:
        mean_metrics, total_metrics = calculate_mean_metrics([result])
        mean_metrics_list.append(mean_metrics)
        total_metrics_list.append(total_metrics)

    # Create a table comparing the metrics for all models
    comparison_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"]
    }

    for idx, metrics in enumerate(mean_metrics_list):
        model_name = f"Model {idx + 1}"
        comparison_data[model_name] = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"]
        ]

    comparison_df = pd.DataFrame(comparison_data)
    print("Comparison Table:")
    print(comparison_df)

    # Generate heatmaps for class-wise metrics of each model
    for idx, metrics in enumerate(mean_metrics_list):
        classwise_df = pd.DataFrame(metrics["class_wise_metrics"]).T
        plt.figure(figsize=(10, 6))
        sns.heatmap(classwise_df, annot=True, fmt=".3f", cmap="coolwarm")
        plt.title(f"Model {idx + 1} - Mean Class-wise Metrics")
        plt.show()

    # Generate QQ plot for accuracy distribution
    for idx, total_metrics in enumerate(total_metrics_list):
        plt.figure(figsize=(6, 6))
        stats.probplot(total_metrics["accuracy"], dist="norm", plot=plt)
        plt.title(f"Model {idx + 1} - QQ Plot for Accuracy")
        plt.show()

    # Generate dummy ROC curve plot (for demonstration, assuming you have TPR and FPR data)
    for idx, total_metrics in enumerate(total_metrics_list):
        # Dummy TPR and FPR for demonstration
        tpr =np.linspace(0, 1, 100) ** (idx + 1)
        fpr = np.linspace(0, 1, 100) ** (idx + 1)  # Just to create a different curve for each model
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Model {idx + 1}')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Model {idx + 1} - ROC Curve')
        plt.legend()
        plt.show()

        table_data = []
        model_names = []
        for idx, metrics in enumerate(mean_metrics_list):
            model_names.append(f'Model {idx + 1}')
            table_data.append([
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"]
            ])

        # Create a DataFrame for easier handling
        metrics_df = pd.DataFrame(
            table_data,
            columns=["Accuracy", "Precision", "Recall", "F1-Score"],
            index=model_names
        )

        # Plot the table
        fig, ax = plt.subplots(figsize=(8, len(metrics_df) + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            rowLabels=metrics_df.index,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.title("Mean Scores for Each Model", pad=20)
        plt.show()
