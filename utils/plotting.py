import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from training.watchers.federated_training_watcher import FederatedTrainingWatcher
from utils.globals import MAX_EPOCHS, NUM_ROUNDS, NUM_CLIENTS

dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {'train_mean': dfc[0], 'train_runs': dfc[0], 'val_mean': dfc[1], 'val_runs': dfc[1]}


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

    plt.subplots_adjust(top=0.85, hspace=0.3, left=0.05, right=0.85)

    # Add a general title for the entire figure
    fig.suptitle('Clients Learning Curves', fontsize=16, y=0.97)


    # Add common column titles 'Client {i}'
    for i in range(NUM_CLIENTS):
        pos = axs[0, i].get_position()
        x_pos = pos.x0 + pos.width / 2
        fig.text(x_pos, 0.85, f'Client {i + 1}', ha='center', va='bottom', fontsize='large')

    legend_labels = ['Training', 'Validation', 'Round Averaging']
    handles = [plt.Line2D([0], [0], color=dfc[0]), plt.Line2D([0], [0], color=dfc[1]), plt.Line2D([0], [0], color='gray', linestyle='--')]
    fig.legend(handles, legend_labels, loc='upper right', fontsize='large', bbox_to_anchor=(0.97, 0.5))

    plt.tight_layout()
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
            scores = round_records[f'{round_num}'][f'{client_id}']
            for key in ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']:
                client_scores[client_id][key].extend(scores[key])


    return client_scores


def stack_client_scores_multiple_runs(all_runs):
    keys = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']

    stacked_client_scores = {}

    for client_id in range(NUM_CLIENTS):
        stacked_client_scores[client_id] = []

    for run_idx in range(len(all_runs)):
        run_scores = all_runs[str(run_idx + 1)]['training_scores']

        # Merge round scores, Keep as is
        client_run_scores = {}
        for client_id in range(NUM_CLIENTS):
            client_run_scores[client_id] = {k: [] for k in keys}

        for round_idx in range(NUM_ROUNDS):
            round_scores = run_scores[str(round_idx)]

            for client_idx in range(NUM_CLIENTS):
                client_round_scores = round_scores[str(client_idx)]

                for key in keys:
                    # EXTEND BY ROUND, SAME CLIENT
                    client_run_scores[client_idx][key].extend(client_round_scores[key])
        # end merge round scores

        # print(f'Run Index: {run_idx + 1}, Merged Run Scores: {client_run_scores}')

        for client_id in range(NUM_CLIENTS):
            # APPEND BY RUN, SAME CLIENT
            stacked_client_scores[client_id].append(client_run_scores[client_id])

    return stacked_client_scores


def plot_runs_mean_with_std(data, loss_or_acc):
    """
    Plot Loss with individual runs, mean, and std as shaded area.

    Args:
        x:
        data: List of dictionaries containing Training Loss and Validation Loss data.
        loss_or_acc:
    Returns:
        None
    """
    assert len(data[0]['Training Loss']) == MAX_EPOCHS
    x = range(0, MAX_EPOCHS)

    mean_training_loss = np.mean([d[f'Training {loss_or_acc}'] for d in data], axis=0)
    std_training_loss = np.std([d[f'Training {loss_or_acc}'] for d in data], axis=0)

    mean_validation_loss = np.mean([d[f'Validation {loss_or_acc}'] for d in data], axis=0)
    std_validation_loss = np.std([d[f'Validation {loss_or_acc}'] for d in data], axis=0)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    # for run in data:
    #     plt.plot(x, run[f'Training {loss_or_acc}'], '--', color=colors['train_runs'], alpha=0.3)
    #     plt.plot(x, run[f'Validation {loss_or_acc}'], '--', color=colors['val_runs'], alpha=0.3)

    plt.plot(x, mean_training_loss, '-', color=colors['train_mean'], label=f'Training {loss_or_acc}')
    plt.fill_between(x,
                     mean_training_loss - std_training_loss,
                     mean_training_loss + std_training_loss,
                     color=colors['train_mean'], alpha=0.2)

    plt.plot(x, mean_validation_loss, '-', color=colors['val_mean'], label=f'Validation {loss_or_acc}')
    plt.fill_between(x,
                     mean_validation_loss - std_validation_loss,
                     mean_validation_loss + std_validation_loss,
                     color=colors['val_mean'], alpha=0.2)

    plt.title(f'{loss_or_acc} Over Epochs (Mean with Std)')
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_or_acc}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_client_runs_mean_with_std(client_ax, x, client_runs, loss_or_acc):
    """
    Plot Loss with individual runs, mean, and std as shaded area, with a simplified legend.

    Args:
        client_ax:
        x:
        client_runs: List of dictionaries containing Training Loss and Validation Loss data.
        loss_or_acc: String whether it's plotting loss or accuracy
        colors: Dictionary to specify colors for mean, runs, and std for training and validation.

    Returns:
        None
    """
    # Extract epochs from client_runs length
    mean_training_loss = np.mean([d[f'Training {loss_or_acc}'] for d in client_runs], axis=0)
    std_training_loss = np.std([d[f'Training {loss_or_acc}'] for d in client_runs], axis=0)
    mean_validation_loss = np.mean([d[f'Validation {loss_or_acc}'] for d in client_runs], axis=0)
    std_validation_loss = np.std([d[f'Validation {loss_or_acc}'] for d in client_runs], axis=0)

    # Plot Loss
    # for run in client_runs:
    #     client_ax.plot(x, run[f'Training {loss_or_acc}'], '--', color=colors['train_runs'], alpha=0.3)
    #     client_ax.plot(x, run[f'Validation {loss_or_acc}'], '--', color=colors['val_runs'], alpha=0.3)

    client_ax.plot(x, mean_training_loss, '-', color=colors['train_mean'], label=f'Training {loss_or_acc}')
    client_ax.fill_between(x,
                           mean_training_loss - std_training_loss,
                           mean_training_loss + std_training_loss,
                           color=colors['train_mean'], alpha=0.2)

    client_ax.plot(x, mean_validation_loss, '-', color=colors['val_mean'], label=f'Validation {loss_or_acc}')
    client_ax.fill_between(x,
                           mean_validation_loss - std_validation_loss,
                           mean_validation_loss + std_validation_loss,
                           color=colors['val_mean'], alpha=0.2)


def plot_clients_learning_curves_multiple_runs(round_scores, num_epochs=MAX_EPOCHS * NUM_ROUNDS):
    stacked_client_scores = stack_client_scores_multiple_runs(round_scores)

    x = range(1, num_epochs + 1)

    fig, axs = plt.subplots(figsize=(4 * NUM_CLIENTS, 9), nrows=2, ncols=NUM_CLIENTS, sharex='col', sharey='row')

    for i, (client_id, client_score) in enumerate(stacked_client_scores.items()):

        # Plot loss curve on the first row
        ax_loss = axs[0, i]
        plot_client_runs_mean_with_std(ax_loss, x, client_score, 'Loss')

        # Plot accuracy curve on the second row
        ax_acc = axs[1, i]
        plot_client_runs_mean_with_std(ax_acc, x, client_score, 'Accuracy')

        # Add vertical dashed lines at positions where i % MAX_EPOCHS == 0
        for epoch in range(MAX_EPOCHS, num_epochs + 1, MAX_EPOCHS):
            ax_loss.axvline(x=epoch, linestyle='--', color='gray')
            ax_acc.axvline(x=epoch, linestyle='--', color='gray')

        ax_loss.xaxis.set_major_locator(ticker.MultipleLocator(base=int(MAX_EPOCHS)))
        ax_acc.xaxis.set_major_locator(ticker.MultipleLocator(base=int(MAX_EPOCHS)))

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

    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.05, right=0.95)

    # Add a general title for the entire figure
    fig.suptitle('Clients Learning Curves', fontsize=16, y=0.97)

    # Add common column titles 'Client {i}'
    for i in range(NUM_CLIENTS):
        pos = axs[0, i].get_position()
        x_pos = pos.x0 + pos.width / 2
        fig.text(x_pos, 0.85, f'Client {i + 1}', ha='center', va='bottom', fontsize='medium')

    legend_labels = ['Training', 'Validation', 'Round Averaging']
    handles = [plt.Line2D([0], [0], color=colors['train_mean']), plt.Line2D([0], [0], color=colors['val_mean']),
               plt.Line2D([0], [0], color='gray', linestyle='--')]
    fig.legend(handles, legend_labels, loc='upper right', fontsize='large', bbox_to_anchor=(0.97, 0.80))

    plt.show()


def plot_server_round_scores(server_round_scores):
    pass