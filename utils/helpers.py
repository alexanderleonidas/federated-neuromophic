import matplotlib.pyplot as plt


def plot_learning_curve(num_epochs, training_scores):
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    x = range(1, num_epochs + 1)
    plt.subplot(1, 2, 1)
    plt.plot(x, training_scores.train_losses, label='Training Loss')
    plt.plot(x, training_scores.valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(x, training_scores.train_accuracies, label='Training Accuracy')
    plt.plot(x, training_scores.valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.show()
