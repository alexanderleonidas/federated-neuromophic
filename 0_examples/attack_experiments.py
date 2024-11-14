from data.mnist_loader import load_mnist_batches
from evaluation.attacks import mia_attack
from models.single_trainable import load_simple_model
from training.single_backprop_training.batch_validation_training import batch_validation_training


def run_mia():
    # USING RESNET-18 ARCHITECTURE

    # batches_dataset = load_mnist_batches(transform=get_augmentation_transform((224, 224)))
    # global_trainable = load_resnet_model(pretrained=False)    # NON PRETRAINED

    #
    # USING A SIMPLE CUSTOM-MADE MODEL


    batches_dataset = load_mnist_batches()
    trainable = load_simple_model()
    untrained = load_simple_model()

    num_epochs = 2
    _ = batch_validation_training(trainable, batches_dataset, num_epochs=num_epochs)

    mia_attack(batches_dataset.train_loader, batches_dataset.test_loader, untrained, trainable)

run_mia()