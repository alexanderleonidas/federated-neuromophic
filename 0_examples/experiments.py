from data.mnist_loader import load_mnist_batches, load_mnist_clients
from evaluation.evaluation import evaluate_outputs
from models.federated.client import Client
from models.federated.server import Server
from models.model_loader import load_simple_model, load_simple_neuromorphic_model
from training.batch_training import batch_validation_training
from training.federated_training.federated_training import federated_training
from training.neuromorphic.neuromorphic_training import neuromorphic_training
from utils.globals import PERTURBATION_BASED, FEEDBACK_ALIGNMENT
from utils.helpers import plot_learning_curve


def run_normal():
    # USING RESNET-18 ARCHITECTURE

    # batches_dataset = load_mnist_batches(transform=get_augmentation_transform((224, 224)))
    # global_model = load_resnet_model(pretrained=False)    # NON PRETRAINED

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
    """Orchestrates the federated training process."""
    num_clients = 3
    rounds = 1
    epochs = 1

    dataset = load_mnist_clients(num_clients)

    trainable_global = load_simple_model()
    server = Server(trainable_global)
    clients = [Client(trainable_global, client_loader) for client_loader in dataset.client_loaders]

    round_results = federated_training(server, clients, rounds=rounds, epochs=epochs)

    metrics = evaluate_outputs(server.global_model.model, dataset.test_loader)
    final_metrics = metrics.get_results()

    print(final_metrics)


def run_neuromorphic_pb():
    method = PERTURBATION_BASED

    batches_dataset = load_mnist_batches()
    trainable = load_simple_neuromorphic_model(method=method)
    num_epochs = 15
    training_scores = neuromorphic_training(trainable, batches_dataset, method=method, num_epochs=num_epochs)
    metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)
    final_metrics = metrics.get_results()

    print(f"Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Precision: {final_metrics['precision']:.2f}%")
    print(f"Recall: {final_metrics['recall']:.2f}%")
    print(f"F1 Score: {final_metrics['f1_score']:.2f}%")

    plot_learning_curve(num_epochs, training_scores)


def run_neuromorphic_fa():
    method = FEEDBACK_ALIGNMENT
    batches_dataset = load_mnist_batches(batch_size=512)
    trainable = load_simple_neuromorphic_model(method=method)
    num_epochs = 5
    training_scores = neuromorphic_training(trainable, batches_dataset, method=method, num_epochs=num_epochs)
    metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)
    final_metrics = metrics.get_results()

    print(f"Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Precision: {final_metrics['precision']:.2f}%")
    print(f"Recall: {final_metrics['recall']:.2f}%")
    print(f"F1 Score: {final_metrics['f1_score']:.2f}%")

    plot_learning_curve(num_epochs, training_scores)


def run_neuromorphic_federated():
    # TODO: This is what we want to explore so not much references I guess?
    pass


run_normal()
# run_normal_federated()
# run_neuromorphic_pb()
# run_neuromorphic_fa()
