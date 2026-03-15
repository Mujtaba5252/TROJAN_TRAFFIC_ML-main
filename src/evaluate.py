import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from src.preprocess import load_data, split_data, scale_data
from src.model import MLP


DATA_PATH = "data/Trojan_Detection_sample.csv"


# load dataset
X, y = load_data(DATA_PATH)

# split dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# scale data
X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

# convert test set to tensor
X_test = torch.tensor(X_test).float()


# load trained model
model = MLP(X_test.shape[1])

model.load_state_dict(torch.load("model.pt"))

model.eval()


with torch.no_grad():

    logits = model(X_test).squeeze()

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).int()


# confusion matrix
cm = confusion_matrix(y_test, preds)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# classification report
print("\nClassification Report:\n")

print(classification_report(y_test, preds))
