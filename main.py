from data.mnist_loader import get_transform, load_data
from evaluation.evaluation import evaluate_model
from models.model_loader import load_simple_model
from training.simple_training import batch_validation_training
from utils.helpers import plot_learning_curve

# TODO: Add state variables: (NORMAL-NEURO) (CLASSIC-FEDERATED)
# TODO: Add model saving accordingly to state

if __name__ == '__main__':

    img_resize = (32, 32)
    transform = get_transform(img_resize)

    train_loader, train_indices, validation_loader, val_indices, test_loader = load_data(
        validation_split=0.2,
        shuffle_dataset=True,
        transform=transform
    )

    # model, criterion, optimizer, scheduler = load_resnet_model(pretrained=False)
    model, criterion, optimizer, scheduler = load_simple_model(img_size=img_resize)

    num_epochs = 2
    train_losses, valid_losses, train_accuracies, valid_accuracies = batch_validation_training(
        model, optimizer, criterion, scheduler,
        train_loader, train_indices, validation_loader, val_indices,
        num_epochs=num_epochs
    )

    evaluate_model(model, test_loader)

    plot_learning_curve(num_epochs, train_losses, train_accuracies, valid_losses, valid_accuracies)

