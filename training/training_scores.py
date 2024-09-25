class TrainingScores:
    def __init__(self, train_losses, valid_losses, train_accuracies, valid_accuracies):
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.train_accuracies = train_accuracies
        self.valid_accuracies = valid_accuracies
