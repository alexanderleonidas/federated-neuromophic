from data.mnist_loader import load_mnist_batches, load_mnist_clients
from evaluation.evaluation import evaluate_outputs
from models.federated_trainable import FederatedTrainable
from models.single_trainable import Trainable
from training.federated_model_trainer import FederatedTrainer
from training.single_model_trainer import Trainer
from utils.globals import pb, fa, NUM_CLIENTS
from utils.plotting import plot_learning_curve, plot_server_round_scores, \
    plot_clients_learning_curves
from utils.state import State

# SINGLE - BACKPROP
def run_normal_single():
    state = State(federated=False, neuromorphic=False, method='backprop', save_model=True)

    batches_mnist_dataset = load_mnist_batches()

    trainable = Trainable(state=state)
    trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)


    trainer.train_model()

    metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

    final_metrics = metrics.get_results()
    metrics.print_results(final_metrics)

    plot_learning_curve(trainer.training_scores)


# SINGLE - BACKPROP w DIFFERENTIAL PRIVACY
def run_normal_single_dp():
    state = State(federated=False, neuromorphic=False, method='backprop-dp', save_model=False)

    batches_mnist_dataset = load_mnist_batches()
    trainable = Trainable(state=state)
    trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)


    trainer.train_model()

    metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

    final_metrics = metrics.get_results()
    metrics.print_results(final_metrics)

    plot_learning_curve(trainer.training_scores)


# FEDERATED - BACKPROP
def run_normal_federated():
    clients_dataset = load_mnist_clients(NUM_CLIENTS)

    state = State(federated=True, fed_type='entire', neuromorphic=False, method='backprop', save_model=True)

    trainable = FederatedTrainable(state=state)
    trainer = FederatedTrainer(trainable=trainable, dataset=clients_dataset, state=state)

    trainer.train_model()

    metrics = evaluate_outputs(trainer.global_model, clients_dataset.test_loader)
    final_metrics = metrics.get_results()
    metrics.print_results(final_metrics)

    plot_clients_learning_curves(trainer.round_scores)
    plot_server_round_scores(trainer.round_scores)


# SINGLE - PERTURBATION BASED
def run_neuromorphic_pb_single():
    method = pb
    state = State(federated=False, neuromorphic=True, method=method)

    batches_dataset = load_mnist_batches()

    trainable = Trainable(state=state)
    trainer = Trainer(trainable= trainable, dataset=batches_dataset, state=state)
    trainer.train_model()

    metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)

    final_metrics = metrics.get_results()
    metrics.print_results(final_metrics)

    plot_learning_curve(trainer.training_scores)


# SINGLE - FEEDBACK ALIGNMENT
def run_neuromorphic_fa_single():
    method = fa
    state = State(federated=False, neuromorphic=True, method=method)

    batches_dataset = load_mnist_batches()

    trainable = Trainable(state=state)
    trainer = Trainer(trainable=trainable, dataset=batches_dataset, state=state)
    trainer.train_model()

    metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)
    final_metrics = metrics.get_results()
    metrics.print_results(final_metrics)

    plot_learning_curve(trainer.training_scores)


# FEDERATED - PERTURBATION BASED
def run_neuromorphic_pb_federated():
    pass

# FEDERATED - FEEDBACK ALIGNMENT
def run_neuromorphic_fa_federated():
    pass

# run_normal_single()
run_normal_federated()
# run_neuromorphic_pb_single()
# run_neuromorphic_fa_single()


