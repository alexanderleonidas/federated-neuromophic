from evaluation.mia.mlLeaks import main_mia_flow
from utils.globals import fa, pb, dp
from utils.state import State
from utils.training_runner_multiple import run_multiple, plot_multiple_runs
import argparse

def runMIA():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topX', type=int, default=3, help='Number of top probabilities to keep')
    opt = parser.parse_args()

    main_mia_flow(
        top_x=opt.topX
    )

if __name__ == '__main__':
    # ss = State(federated=False, neuromorphic=False, method=dp,  save_model=True)
    # fp, xcore, idx = run_multiple(ss)

    #plot_multiple_runs(ss.federated, xcore)
    runMIA()

