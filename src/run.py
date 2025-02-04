from experiment import experiment


class Run:
    def __init__(self, model: str, system: str, verbose: str):
        self.model = model if self.__check_model__(model) else None
        self.system = system if self.__check_system__(system) else None
        self.verbose = verbose if verbose else False

    def run_exp(self):
        exp_dict = {'model': self.model, 'system': self.system, 'verbose': self.verbose}
        experiment(exp_dict)

    def set_dp_parameters(self):
        # Differential Privacy Parameters
        self.epsilon = 0.1
        self.delta = 2e-5

    @staticmethod
    def __check_model__(model: str) -> bool:
        allowed_models = ['dp', 'dfa', 'npb', 'wpb']
        if model.lower() in allowed_models:
            return True
        else:
            raise Exception('Invalid model string.'
                            'Accepted: DP or DFA, NPB, WPB')

    @staticmethod
    def __check_system__(system: str) -> bool:
        if system == 'CL' or "FL":
            return True
        else:
            raise Exception('Invalid system string.'
                            'Accepted: CL or FL')