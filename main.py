from utils.globals import pb
from utils.state import State
from utils.training_runner import run_single_model

if __name__ == '__main__':
    state = State(neuromorphic=True, method=pb)
    run_single_model(state)
