import matplotlib.pyplot as plt

from utils.globals import MAX_EPOCHS

dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss_curve(ax, x, training_scores):
    ax.plot(x, training_scores['Training Loss'], label='Training Loss', color=dfc[0])
    ax.plot(x, training_scores['Validation Loss'], label='Validation Loss', color=dfc[1])

def plot_accuracy_curve(ax, x, training_scores):
    ax.plot(x, training_scores['Training Accuracy'], label='Training Accuracy', color=dfc[0])
    ax.plot(x, training_scores['Validation Accuracy'], label='Validation Accuracy', color=dfc[1])

def plot_learning_curve(training_scores, num_epochs=MAX_EPOCHS):
    x = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    ax = plt.subplot(1, 2, 1)
    # Plot loss curves
    plot_loss_curve(ax, x, training_scores)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy / Epoch'); plt.legend()

    ax = plt.subplot(1, 2, 2)
    # Plot accuracy curves
    plot_accuracy_curve(ax, x, training_scores)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss / Epoch'); plt.legend()

    plt.tight_layout()
    plt.show()

def plot_clients_learning_curves(clients_training_scores, num_epochs=MAX_EPOCHS):
    x = range(1, num_epochs + 1)

    num_clients = len(clients_training_scores)
    if num_clients == 1:
        return plot_learning_curve(clients_training_scores[0])

    fig, axs = plt.subplots(2, num_clients, sharex='col', sharey='row')

    for i, client_score in enumerate(clients_training_scores):
        # Plot loss curve on the first row
        ax_loss = axs[0, i]
        plot_loss_curve(ax_loss, x, client_score)

        # Plot accuracy curve on the second row
        ax_acc = axs[1, i]
        plot_accuracy_curve(ax_acc, x, client_score)

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

    plt.subplots_adjust(top=0.80, hspace=0.4)

    # Add a general title for the entire figure
    fig.suptitle('Clients Learning Curves', fontsize=16, y=0.97)


    # Add common column titles 'Client {i}'
    for i in range(num_clients):
        pos = axs[0, i].get_position()
        x_pos = pos.x0 + pos.width / 2
        fig.text(x_pos, 0.85, f'Client {i + 1}', ha='center', va='bottom', fontsize='large')

    legend_labels = ['Training', 'Validation']
    handles = [plt.Line2D([0], [0], color=dfc[0]), plt.Line2D([0], [0], color=dfc[1])]

    # Add legend to the first row (Loss)
    axs[0, num_clients - 1].legend(handles, legend_labels, loc='upper right')

    # Add legend to the second row (Accuracy)
    axs[1, num_clients - 1].legend(handles, legend_labels, loc='upper right')

    plt.show()

def plot_server_round_scores(server_round_scores):
    pass