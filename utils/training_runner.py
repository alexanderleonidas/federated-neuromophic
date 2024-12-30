from data.mnist_loader import load_mnist_batches, load_mnist_clients
from evaluation.evaluation import evaluate_outputs
from models.federated_trainable import FederatedTrainable
from models.single_trainable import Trainable
from training.federated_model_trainer import FederatedTrainer
from training.single_model_trainer import Trainer
from utils.globals import pb, fa, NUM_CLIENTS, VERBOSE
from utils.plotting import plot_learning_curve, plot_server_round_scores, plot_clients_learning_curves
from utils.state import State


# SINGLE MODEL - BACKPROP BY DEFAULT
def run_single_model(state=None):
    if state is None:  state = State()

    # load dataset
    batches_dataset = load_mnist_batches()

    # load model_type components
    trainable = Trainable(state=state)
    if state.method == 'backprop-dp':   trainable.support_dp_engine(batches_dataset)

    # set-up and training
    trainer = Trainer(trainable=trainable, dataset=batches_dataset, state=state)
    trainer.train_model()

    # evaluate on test set
    test_metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)
    final_metrics = test_metrics.get_results()

    # log
    if VERBOSE:
        test_metrics.print_results(final_metrics)
        plot_learning_curve(trainer.training_scores)

    # return test results and training recordings
    return final_metrics, trainer.training_scores

# SINGLE - BACKPROP w DIFFERENTIAL PRIVACY
def run_normal_single_dp():
    state = State(federated=False, neuromorphic=False, method='backprop-dp')
    return run_single_model(state)

# SINGLE - by method
def run_neuromorphic_single(method):
    state = State(federated=False, neuromorphic=True, method=method)
    return run_single_model(state)

# SINGLE - PERTURBATION BASED
def run_neuromorphic_pb_single():
    method = pb
    return run_neuromorphic_single(method)

# SINGLE - FEEDBACK ALIGNMENT
def run_neuromorphic_fa_single():
    method = fa
    return run_neuromorphic_single(method)

#  v   FEDERATED MODEL RUNNING   v

# FEDERATED - BACKPROP BY DEFAULT
def run_federated_training(state):
    if not state:
        state = State(federated=True, fed_type='entire')

    clients_dataset = load_mnist_clients(NUM_CLIENTS)

    trainable = FederatedTrainable(state=state)
    trainer = FederatedTrainer(trainable=trainable, dataset=clients_dataset, state=state)

    trainer.train_model()

    metrics = evaluate_outputs(trainer.global_model, clients_dataset.test_loader)
    final_metrics = metrics.get_results()

    if VERBOSE:
        metrics.print_results(final_metrics)
        plot_clients_learning_curves(trainer.round_scores)
        plot_server_round_scores(trainer.round_scores)

    return metrics.get_results(), trainer.round_scores

# FEDERATED - BACKPROP with DIFFERENTIAL PRIVACY
def run_back_dp_federated():
    state = State(federated=True, fed_type='entire', method='backprop_dp')
    return run_federated_training(state)

# FEDERATED - PERTURBATION BASED
def run_neuromorphic_pb_federated():
    state = State(federated=True, fed_type='entire', neuromorphic=True, method=pb)
    return run_federated_training(state)

# FEDERATED - FEEDBACK ALIGNMENT
def run_neuromorphic_fa_federated():
    state = State(federated=True, fed_type='entire', neuromorphic=True, method=fa)
    return run_federated_training(state)


# run_normal_single()
# run_normal_single_dp()
# run_normal_federated()
# run_neuromorphic_pb_single()
# run_neuromorphic_fa_single()
# run_neuromorphic_pb_federated()
# run_neuromorphic_fa_federated()
