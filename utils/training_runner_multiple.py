import json
import os

from training_runner import *
from utils.globals import TARGET_EPSILON as eps
from utils.plotting import plot_clients_learning_curves_multiple_runs, plot_runs_mean_with_std

NUM_RUNS = 10
results_path = './results'


def get_file_path_by_state(state: State):
    federated_or_single = 'federated' if state.federated else 'single'

    method = state.method
    if state.method == fa: method = 'fa'
    elif state.method == pb: method = 'pb'
    elif method == 'backprop-dp': method = f'back_dp_{str(eps).replace('.', 'p')}'

    return f'{results_path}/training_scores_{federated_or_single}_{method}_{NUM_RUNS}runs.json'

def load_json_data(file_path):
    # Function to load JSON
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Returning an empty list.")
        return []
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}. Returning an empty list.")
            return []

def load_recorded_scores(state):
    if not os.path.exists(results_path): os.makedirs(results_path)

    file_path = get_file_path_by_state(state)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)

    scores = load_json_data(file_path)
    rec_scores = list(scores.keys())
    # if there is any existing run recorded, keep it and scroll index
    last_it = rec_scores[-1] if len(rec_scores) > 0 else 0

    return file_path, scores, last_it

def get_federated_model_running_fn(state):
    if state.neuromorphic:
        if state.method == pb: return run_neuromorphic_pb_federated
        elif state.method == fa: return run_neuromorphic_fa_federated
    else:
        if state.method == 'back_dp':  return run_back_dp_federated
        elif state.method == 'backprop':return run_federated_training

def run_multiple_federated(state):
    assert state.federated
    training_scores_file, loaded_scores, last_iteration = load_recorded_scores(state)
    training_fn = get_federated_model_running_fn(state)

    all_training_scores = []
    all_final_metrics = []

    for i in range(NUM_RUNS):
        final_metrics, training_scores = training_fn()
        del final_metrics['confusion_matrix']
        all_final_metrics.append(final_metrics)
        all_training_scores.append(training_scores)

        # Prepare the client_runs with iteration count
        scores = {
            'final_metrics': final_metrics,
            'training_scores': training_scores.round_records
        }

        it_num = int(last_iteration) + i + 1
        iteration_key = str(it_num)
        print(f'Run Number {it_num} >>')

        # Update final_metrics.json
        with open(training_scores_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            data[iteration_key] = scores

            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()  # Remove any leftover client_runs

    loaded_scores = load_json_data(training_scores_file)
    plot_clients_learning_curves_multiple_runs(loaded_scores)

def get_single_model_running_fn(state):
    if state.neuromorphic:
        if state.method == pb: return run_neuromorphic_pb_single
        elif state.method == fa: return run_neuromorphic_fa_single
    else:
        if state.method == 'back_dp': return run_normal_single_dp
        elif state.method == 'backprop': return run_single_model

def run_multiple_single_model(state):
    assert not state.federated

    training_scores_file, loaded_scores, last_iteration = load_recorded_scores(state)
    training_fn = get_single_model_running_fn(state)

    all_training_scores = []
    all_final_metrics = []

    for i in range(NUM_RUNS):
        final_metrics, training_scores = training_fn()
        del final_metrics['confusion_matrix']
        all_final_metrics.append(final_metrics)
        all_training_scores.append(training_scores)

        # Prepare the client_runs with iteration count
        scores = {
            'final_metrics': final_metrics,
            'training_scores': training_scores
        }

        it_num = int(last_iteration) + i + 1
        iteration_key = str(it_num)
        print(f'Run Number {it_num} >>')

        # Update final_metrics.json
        with open(training_scores_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            data[iteration_key] = scores

            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()  # Remove any leftover client_runs

    loaded_scores = load_json_data(training_scores_file)
    plot_runs_mean_with_std([sc['training_scores'] for _, sc in loaded_scores.items()], 'Accuracy')

def run_multiple(state):
    if state.federated: run_multiple_federated(state)
    else: run_multiple_single_model(state)
