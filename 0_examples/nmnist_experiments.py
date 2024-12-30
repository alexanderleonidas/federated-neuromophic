from data.neuro_mnist_loader import load_n_mnist_batches
from evaluation.evaluation import evaluate_outputs
from models.single_trainable import Trainable
from training.single_model_trainer import Trainer
from utils.plotting import plot_learning_curve
from utils.state import State


# SINGLE MODEL - BACKPROP BY DEFAULT
def test_nmnist_training():
    state = State()  # default single model with backprop
    state.model_type = 'snn'

    # load dataset
    dataset = load_n_mnist_batches()

    # load model_type components
    trainable = Trainable(state=state)

    # set-up and training
    trainer = Trainer(trainable=trainable, dataset=dataset, state=state)
    trainer.train_model()

    # evaluate on test set
    test_metrics = evaluate_outputs(trainable.model, dataset.test_loader)
    final_metrics = test_metrics.get_results()

    # log
    test_metrics.print_results(final_metrics)
    plot_learning_curve(trainer.training_scores)

    # return test results and training recordings
    return final_metrics, trainer.training_scores
