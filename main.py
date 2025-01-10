from utils.globals import fa
from utils.state import State
from utils.training_runner import run_single_model, run_federated_model

if __name__ == '__main__':
    state = State(neuromorphic=True, federated=True, method=fa)
    run_federated_model(state)
