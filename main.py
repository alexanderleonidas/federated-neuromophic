from utils.globals import fa, pb
from utils.state import State
from utils.training_runner_multiple import run_multiple, plot_multiple_runs

if __name__ == '__main__':
    ss = State(federated=True, neuromorphic=True, method=fa,  save_model=True)
    fp, xcore, idx = run_multiple(ss)

    #plot_multiple_runs(ss.federated, xcore)
