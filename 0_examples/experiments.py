import numpy as np

from data.mnist_loader import load_mnist_batches
from evaluation.evaluation import evaluate_outputs
from evaluation.mia import membership_inference_attack
from models.single_trainable import Trainable
from training.single_model_trainer import Trainer
from utils.globals import fa
from utils.state import State
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

# TODO: this to re-usable code for experiments
# TODO: adapt to federated? Which client owns which data?

state = State(neuromorphic=True, method=fa)

# load dataset
batches_dataset = load_mnist_batches()

# load model components
trainable = Trainable(state=state)
trainer = Trainer(trainable=trainable, dataset=batches_dataset, state=state)

# train model
trainer.train_model()

# evaluate on test set
test_metrics = evaluate_outputs(trainable.model, batches_dataset.test_loader)
final_metrics = test_metrics.get_results()

# run MIA
mia_labels, mia_preds = membership_inference_attack(trainable, batches_dataset)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Convert to numpy arrays
all_labels = np.array(mia_labels)
all_preds = np.array(mia_preds)

# Binary predictions
binary_preds = (mia_preds > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(mia_labels, binary_preds)
roc_auc = roc_auc_score(mia_labels, mia_preds)
cr = classification_report(mia_labels, binary_preds, target_names=['Non-member', 'Member'], zero_division=0.)

print(f'Attack Model Accuracy: {accuracy:.4f}\n')
print(f'Attack Model ROC AUC: {roc_auc:.4f}\n')
print(f'Classification Report: \n{cr}\n')

cm = confusion_matrix(mia_labels, binary_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Attack Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Distribution of Predicted Probabilities
member_preds = mia_preds[mia_labels == 1]
nonmember_preds = mia_preds[mia_labels == 0]

plt.figure(figsize=(12, 6))
sns.kdeplot(member_preds.flatten(), label='Members', fill=True)
sns.kdeplot(nonmember_preds.flatten(), label='Non-members', fill=True)
plt.title('Distribution of Attack Model Predictions')
plt.xlabel('Predicted Probability of Being a Member')
plt.ylabel('Density')
plt.legend()
plt.show()


# ROC Curve
fpr, tpr, thresholds = roc_curve(mia_labels, mia_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
