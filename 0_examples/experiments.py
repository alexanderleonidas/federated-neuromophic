from data.mnist_loader import load_mnist_batches, load_mnist_clients
from evaluation.evaluation import evaluate_outputs, compare_models
from models.federated_trainable import FederatedTrainable
from models.single_trainable import Trainable
from training.federated_model_trainer import FederatedTrainer
from training.single_model_trainer import Trainer
from training.single_neuromorphic_training.perturbation_based import evaluate_best_p_std
from utils.globals import pb, fa, NUM_CLIENTS, PATH_TO_SAVED_RESULTS, NUM_EXPERIMENTS
from utils.plotting import plot_learning_curve, plot_server_round_scores, \
    plot_clients_learning_curves, evaluation_metrics
from utils.state import State
from data.experiment_data.JSON.json_manager import JSONManager


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
def run_normal_federated(state=None):
    clients_dataset = load_mnist_clients(NUM_CLIENTS)

    if state is None:
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
    trainer = Trainer(trainable=trainable, dataset=batches_dataset, state=state)
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


################# Comparing Models - Benchmarking  #################

def run_normal_vs_neuromorphic_single(normal_results_fp, neuromorphic_results_fp, NUM_EXPERIMENTS=NUM_EXPERIMENTS):
    # load or download MNIST dataset
    batches_mnist_dataset = load_mnist_batches()

    normal_results = []
    neuromorphic_results = []

    for i in range(NUM_EXPERIMENTS):
        # normal
        state = State(federated=False, neuromorphic=False, method='backprop', save_model=True)

        trainable = Trainable(state=state)
        trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)

        trainer.train_model()

        metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

        normal_final_results = metrics.get_results()
        print(f"Experiment {i} - Normal Training")
        normal_results.append(normal_final_results)

        # neuromorphic
        state = State(federated=False, neuromorphic=True, method=fa, save_model=True)

        batches_mnist_dataset = load_mnist_batches()
        trainable = Trainable(state=state)
        trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)

        trainer.train_model()

        metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

        neuromorphic_final_results = metrics.get_results()
        print(f"Experiment {i} - Neuromorphic Training")
        neuromorphic_results.append(neuromorphic_final_results)

        if (i + 1) % 5 == 0:
            print(f"Preliminary Evaluation {i / 5}- Experiment {i + 1} ")
            normal_accuracies = [m['precision'] for m in normal_results]
            neuromorphic_accuracies = [m['precision'] for m in neuromorphic_results]
            print("neuromorphic_accuracies ", neuromorphic_accuracies)
            print("normal_accuracies", normal_accuracies)

            print("Evaluating Normal vs Neuromorphic Single Training")
            compare_models(normal_accuracies, neuromorphic_accuracies)

    normal_accuracies = [m['precision'] for m in normal_results]
    neuromorphic_accuracies = [m['precision'] for m in neuromorphic_results]
    print("neuromorphic_accuracies ", neuromorphic_accuracies)
    print("normal_accuracies", normal_accuracies)

    print("Evaluating Normal vs Neuromorphic Single Training")
    compare_models(normal_accuracies, neuromorphic_accuracies)
    for i in range(NUM_EXPERIMENTS):
        normal_results[i]['confusion_matrix'] = 0
        neuromorphic_results[i]['confusion_matrix'] = 0

    json_manager = JSONManager()

    json_manager.save_to_json(normal_results,
                              normal_results_fp)
    json_manager.save_to_json(neuromorphic_results,
                              neuromorphic_results_fp)

    print("Metrics saved to json")


def run_normal_vs_neuromorphic_federated(normal_results_fp, neuromorphic_results_fp, NUM_EXPERIMENTS=NUM_EXPERIMENTS):
    # load or download MNIST dataset
    batches_mnist_dataset = load_mnist_batches()

    normal_results = []
    neuromorphic_results = []

    for i in range(NUM_EXPERIMENTS):
        # normal
        state = State(federated=True, neuromorphic=False, method='backprop', save_model=True)

        trainable = Trainable(state=state)
        trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)

        trainer.train_model()

        metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

        normal_final_results = metrics.get_results()
        print(f"Experiment {i} - Normal Training")
        normal_results.append(normal_final_results)

        # neuromorphic
        state = State(federated=True, neuromorphic=True, method=fa, save_model=True)

        batches_mnist_dataset = load_mnist_batches()
        trainable = Trainable(state=state)
        trainer = Trainer(trainable=trainable, dataset=batches_mnist_dataset, state=state)

        trainer.train_model()

        metrics = evaluate_outputs(trainable.model, batches_mnist_dataset.test_loader)

        neuromorphic_final_results = metrics.get_results()
        print(f"Experiment {i} - Neuromorphic Training")
        neuromorphic_results.append(neuromorphic_final_results)

        if (i + 1) % 5 == 0:
            print(f"Preliminary Evaluation {i / 5}- Experiment {i + 1} ")
            normal_accuracies = [m['precision'] for m in normal_results]
            neuromorphic_accuracies = [m['precision'] for m in neuromorphic_results]
            print("neuromorphic_accuracies ", neuromorphic_accuracies)
            print("normal_accuracies", normal_accuracies)

            print("Evaluating Normal vs Neuromorphic Single Training")
            compare_models(normal_accuracies, neuromorphic_accuracies)

    normal_accuracies = [m['precision'] for m in normal_results]
    neuromorphic_accuracies = [m['precision'] for m in neuromorphic_results]
    print("neuromorphic_accuracies ", neuromorphic_accuracies)
    print("normal_accuracies", normal_accuracies)

    print("Evaluating Normal vs Neuromorphic Single Training")
    compare_models(normal_accuracies, neuromorphic_accuracies)
    for i in range(NUM_EXPERIMENTS):
        normal_results[i]['confusion_matrix'] = 0
        neuromorphic_results[i]['confusion_matrix'] = 0

    json_manager = JSONManager()

    json_manager.save_to_json(normal_results,
                              normal_results_fp)
    json_manager.save_to_json(neuromorphic_results,
                              neuromorphic_results_fp)

    print("Metrics saved to json")


# FEDERATED - PERTURBATION BASED
def run_neuromorphic_pb_federated():
    pass


# FEDERATED - FEEDBACK ALIGNMENT
def run_neuromorphic_fa_federated():
    state = State(federated=True, fed_type='entire', neuromorphic=False, method=fa, save_model=True)
    run_normal_federated(state)


def run_evaluation():
    normal_results_fp = 'normal_single_results.json'
    neuromorphic_results_fp =  'neuromorphic_single_results.json'

    json_manager = JSONManager()
    normal_results = json_manager.load_from_json(normal_results_fp)
    neuromorphic_results = json_manager.load_from_json(neuromorphic_results_fp)

    results = {
        'normal': normal_results,
        'neuromorphic': neuromorphic_results
    }
    evaluation_metrics(results)

# run_normal_single()
# run_normal_federated()
#run_neuromorphic_pb_single()
# run_neuromorphic_fa_single()

#run_normal_vs_neuromorphic_single('normal_backprop_single_results_10epochs_standardtrainingparameters.json','neuromorphic_FA_single_results_10epochs_standardtrainingparameters.json')

#un_neuromorphic_fa_federated()

run_evaluation()
