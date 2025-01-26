from evaluation.mia.mlLeaks import main_mia_flow
from utils.globals import fa, pb, dp, bp
from utils.state import State
from utils.training_runner import runMIA, run_single_model, run_federated_model
from utils.training_runner_multiple import run_multiple, plot_multiple_runs, run_multiple_MIA
import argparse



if __name__ == '__main__':
    # ss = State(federated=False, neuromorphic=False, method=bp,  save_model=True)
    # fp, xcore, idx = run_multiple(ss)

    #plot_multiple_runs(ss.federated, xcore)
    state = State(federated=True, neuromorphic=True, method=fa,  save_model=True)
    run_multiple_MIA(state)

