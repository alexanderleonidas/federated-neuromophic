import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

from data.mnist_loader import load_mnist_clients
from evaluation.evaluation import evaluate_outputs
from evaluation.old_mia.mia import membership_inference_attack
from models.federated_trainable import FederatedTrainable
from training.federated_model_trainer import FederatedTrainer
from utils.state import State

# TODO: adapt to federated? Which client owns which data?

state = State(neuromorphic=False, federated=True, method='backprop')

# In the federated case >>>>>>>>>>>> #################################################

# load dataset
batches_dataset = load_mnist_clients()

# load model_type components
trainable = FederatedTrainable(state=state)
trainer = FederatedTrainer(trainable=trainable, dataset=batches_dataset, state=state)

######################################################################################

# train model_type
trainer.train_model()

# evaluate on test set
test_metrics = evaluate_outputs(trainer.global_model, batches_dataset.test_loader)
final_metrics = test_metrics.get_results()

# run mia
mia_labels, mia_preds = membership_inference_attack(trainable, batches_dataset)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Convert to numpy arrays
all_labels = np.array(mia_labels)
all_preds = np.array(mia_preds)

# Binary predictions
binary_preds = (all_preds > 0.5).astype(int)

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
member_preds = all_preds[mia_labels == 1]
nonmember_preds = all_preds[mia_labels == 0]

plt.figure(figsize=(12, 6))
sns.kdeplot(member_preds.flatten(), label='Members', fill=True)
sns.kdeplot(nonmember_preds.flatten(), label='Non members', fill=True)
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
