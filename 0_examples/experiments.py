from data.mnist_loader import get_augmentation_transform, load_mnist_batches
from evaluation.evaluation import evaluate_outputs
from models.model_loader import load_simple_model, load_resnet_model
from training.batch_training import batch_validation_training
from utils.helpers import plot_learning_curve
from utils.state import State


def run_normal():
    # USING RESNET-18 ARCHITECTURE

    # batches_dataset = load_mnist_batches(transform=get_augmentation_transform((224, 224)))
    # trainable = load_resnet_model(pretrained=False)    # NON PRETRAINED

    #
    # USING A SIMPLE CUSTOM-MADE MODEL

    batches_dataset = load_mnist_batches()
    trainable = load_simple_model()

    num_epochs = 2
    training_scores = batch_validation_training(trainable, batches_dataset, num_epochs=num_epochs)

    metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)

    final_metrics = metrics.get_results()

    print(f"Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Precision: {final_metrics['precision']:.2f}%")
    print(f"Recall: {final_metrics['recall']:.2f}%")
    print(f"F1 Score: {final_metrics['f1_score']:.2f}%")

    plot_learning_curve(num_epochs, training_scores)


def run_normal_federated():
    # TODO: at least this should be easy doable
    pass


def run_neuromorphic():
    # TODO: for this we need some references
    pass


def run_neuromorphic_federated():
    # TODO: This is what we want to explore so not much references I guess?
    pass


run_normal()
