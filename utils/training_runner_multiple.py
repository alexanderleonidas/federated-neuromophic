import json
import os

from utils.training_runner import *
from utils.plotting import plot_clients_learning_curves_multiple_runs, plot_runs_mean_with_std

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_RUNS = 3      # The total num of runs : (NUM_RUNS - 1) *actually in practice
# the total runs that you want to run
# if the file has already existing runs, then the start will be from the last iteration,
# until NUM_RUNS is reached
# it can also be -1 to load all existing and count that as the total

MAX_RUNS_FED = 50           # mostly to stick to file naming being the same
MAX_RUNS_SINGLE = 100

results_path = './results'  # results directory

extra_suffix = ''                  # if you have extra parameters you want to test for the same combination
# extra_suffix = '_3c10e3r'        # example for 3 clients 10 epochs 3 rounds
# extra_suffix = '_3c1e9r'         # example for 3 clients 1 epoch 9 rounds

OVERWRITE_EXISTING = False         # if you want to override what you have saved

# IMPORTANT !!!
# for any configuration, global variables have to match at all times with saved results
# 1 when doing multiple runs saving, 2 when loading back e.g. to make plots, etc...

# at all runs the json structure should stay the same, hence num clients, num rounds, num epochs etc...
# should also match between all runs (mostly applies to the federated case)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_file_path_by_state(state: State):
    if state.federated:
        fs = 'federated'
        nr = MAX_RUNS_FED
    else:
        fs = 'single'
        nr = MAX_RUNS_SINGLE

    method = state.method
    if state.method == fa: method = 'fa'
    elif state.method == pb: method = 'pb'

    return f'{results_path}/training_scores_{fs}_{method}_{nr}runs{extra_suffix}.json'

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

    print(f'Loading scores from {file_path}')

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Trying to create new results?")
        str_in = input('[Y/N]')

        if str_in.lower() == 'n':
            exit(0)

        with open(file_path, 'w') as f:
            json.dump({}, f)

    scores = load_json_data(file_path)
    rec_scores = list(scores.keys())
    # if there is any existing run recorded, keep it and scroll index
    last_it = int(rec_scores[-1]) if len(rec_scores) > 0 else 0

    print(f'Loaded {last_it} scores from {file_path}')
    return file_path, scores, last_it

def get_federated_model_running_fn(state):
    if state.neuromorphic:
        if state.method == pb: return run_neuromorphic_pb_federated
        elif state.method == fa: return run_neuromorphic_fa_federated
    else:
        if state.method == 'back_dp':  return run_back_dp_federated
        elif state.method == 'backprop':return run_federated_model

def run_and_save_multiple_iterations(training_scores_file, loaded_scores, last_iteration, state, extract_results):
    all_training_scores = []
    all_final_metrics = []

    if NUM_RUNS <= 0:
        iterations = len(loaded_scores)
    else:
        iterations = NUM_RUNS

    if OVERWRITE_EXISTING:
        print(f'You are going to override all existing results in file {training_scores_file}')
        print(f'ARE YOU SURE? Yes will start from 0, No will exit the program')
        str_in = input('[Y/N]')
        if str_in.replace(' ', '').lower() == 'y':
            first_iteration = 0
        else:
            exit(0)
    else:
        first_iteration = last_iteration + 1
        if first_iteration > iterations:
            first_iteration = iterations

    print(f'On run from iteration {first_iteration}, up to {iterations}')

    for i in range(first_iteration, iterations):
        print(f'Starting iteration {i}')
        if state.federated:
            final_metrics, training_scores = run_federated_model(state)
        else:
            final_metrics, training_scores = run_single_model(state)
        del final_metrics['confusion_matrix']
        all_final_metrics.append(final_metrics)
        all_training_scores.append(training_scores)

        # Prepare the client_runs with iteration count
        scores = {
            'final_metrics': final_metrics,
            'training_scores': extract_results(training_scores)
        }

        iteration_key = str(i)
        print(f'Saving in Results: Run Number {i}/{iterations} >>')

        # Update final_metrics.json
        with open(training_scores_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            data[iteration_key] = scores

            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

def get_single_model_running_fn(state):
    if state.neuromorphic:
        if state.method == pb: return run_neuromorphic_pb_single
        elif state.method == fa: return run_neuromorphic_fa_single
    else:
        if state.method == 'back_dp': return run_normal_single_dp
        elif state.method == 'backprop': return run_single_model

def run_multiple_federated(state):
    assert state.federated
    training_scores_file, loaded_scores, last_iteration = load_recorded_scores(state)
    extract_results = lambda x: x.round_records

    run_and_save_multiple_iterations(training_scores_file, loaded_scores, last_iteration, state, extract_results)

    return training_scores_file, loaded_scores, last_iteration

def run_multiple_single_model(state):
    assert not state.federated
    training_scores_file, loaded_scores, last_iteration = load_recorded_scores(state)
    extract_results = lambda x: x

    run_and_save_multiple_iterations(training_scores_file, loaded_scores, last_iteration, state, extract_results)

    return training_scores_file, loaded_scores, last_iteration

def run_multiple(state):
    return run_multiple_federated(state) if state.federated else run_multiple_single_model(state)

def plot_multiple_runs(federated, loaded_scores):
    if federated:
        plot_clients_learning_curves_multiple_runs(loaded_scores)
    else:
        plot_runs_mean_with_std([sc['training_scores'] for _, sc in loaded_scores.items()], 'Accuracy')


# example usage
# ss = State(federated=False, neuromorphic=False, method='backprop')
# fp, xcore, idx = run_multiple(ss)
# plot_multiple_runs(ss.federated, xcore)
