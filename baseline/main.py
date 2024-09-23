from baseline.simple_training import batch_validation_training, evaluation, plot_learning_curve
from baseline.mnist_loader import load_data, get_transform
from baseline.model_loader import load_resnet_model, load_simple_model

if __name__ == '__main__':

    img_resize = (64, 64)
    transform = get_transform(img_resize)

    train_loader, train_indices, validation_loader, val_indices, test_loader = load_data(
        validation_split=0.2,
        shuffle_dataset=True,
        transform=transform
    )

    # model, criterion, optimizer, scheduler = load_resnet_model(pretrained=False)
    model, criterion, optimizer, scheduler = load_simple_model(img_size=img_resize)

    num_epochs = 6
    train_losses, valid_losses, train_accuracies, valid_accuracies = batch_validation_training(
        model, optimizer, criterion, scheduler,
        train_loader, train_indices, validation_loader, val_indices,
        num_epochs=num_epochs
    )

    evaluation(model, test_loader)

    plot_learning_curve(num_epochs, train_losses, train_accuracies, valid_losses, valid_accuracies)
