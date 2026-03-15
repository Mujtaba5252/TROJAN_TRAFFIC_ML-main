import torch
import numpy as np
import torch.profiler

from src.preprocess import load_data, split_data, scale_data
from src.model import MLP

from sklearn.metrics import roc_auc_score


DATA_PATH = "data/Trojan_Detection_sample.csv"

# reproducibility
np.random.seed(42)
torch.manual_seed(42)


# load dataset
X, y = load_data(DATA_PATH)

print("Dataset shape:", X.shape)

# split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# scale features
X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

# convert to tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train.values).float()

X_val = torch.tensor(X_val).float()
y_val = torch.tensor(y_val.values).float()


# create model
model = MLP(X_train.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.BCEWithLogitsLoss()


with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True
) as prof:

    for epoch in range(20):

        model.train()

        logits = model(X_train).squeeze()

        loss = criterion(logits, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(X_val).squeeze()

            preds = torch.sigmoid(val_logits)

            auc = roc_auc_score(y_val.numpy(), preds.numpy())

        print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val AUROC {auc:.4f}")


print("\nProfiler results:\n")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


# save model
torch.save(model.state_dict(), "model.pt")

print("\nModel saved as model.pt")
