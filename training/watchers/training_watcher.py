class TrainingWatcher:
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.best_val_acc = 0.0
        self.should_save = False


    def record_epoch(self, train_loss, train_accuracy, valid_loss, valid_accuracy):
        self.record_batch(train_loss, train_accuracy)
        self.record_validation(valid_loss, valid_accuracy)

    def record_batch(self, training_loss, training_accuracy):
        self.train_losses.append(training_loss)
        self.train_accuracies.append(training_accuracy)

    def record_validation(self, validation_loss, validation_accuracy):
        self.valid_losses.append(validation_loss)
        self.valid_accuracies.append(validation_accuracy)

        if validation_accuracy > self.best_val_acc:
            self.best_val_acc = validation_accuracy
            self.should_save = True

    def is_best_accuracy(self):
        var = self.should_save
        self.should_save = False        # for next iteration
        return var
      
    def get_records(self):
        return {
            'Training Loss': self.train_losses,
            'Training Accuracy': self.train_accuracies,
            'Validation Loss': self.valid_losses,
            'Validation Accuracy': self.valid_accuracies
        }
