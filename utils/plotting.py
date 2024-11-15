import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from training.watchers.federated_training_watcher import FederatedTrainingWatcher
from utils.globals import MAX_EPOCHS, NUM_ROUNDS, NUM_CLIENTS

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