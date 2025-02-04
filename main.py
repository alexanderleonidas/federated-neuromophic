from src.run import Run
#############################################
##                                         ##
##### Details on how to run experiments #####
##                                         ##
#############################################

### 1. Choose Experiment parameters
NUM_EXPERIMENTS = 10

# 2. Choose Centralised ('CL') or Federated ('FL') Learning
SYSTEM = 'CL'
# SYSTEM = 'FL'

# 3. Choose model type from:
# Backpropagation with Differential Privacy: DP
# Direct Feedback Alignment: DFA
# Perturbation-based Learning: (weight) WPB, or (node) NPB
MODEL = 'DP'
# MODEL = 'DFA'
# MODEL = 'WPB'
# MODEL = 'NPB'

# 4. Perform the experiment
Run(MODEL, )
