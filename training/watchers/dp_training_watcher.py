from training.watchers.training_watcher import TrainingWatcher


class TrainingWatcherDP(TrainingWatcher):
    def __init__(self):
        super().__init__()
        self.privacy_spent = []

    def record_epoch_dp(self, train_loss, train_accuracy, valid_loss, valid_accuracy, privacy_spent):
        super().record_epoch(train_loss, train_accuracy, valid_loss, valid_accuracy)
        self.privacy_spent.append(privacy_spent)
