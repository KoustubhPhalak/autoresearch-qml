import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# iter5: Pure StatePrep → qml.probs(), no trainable quantum layers
# The computational basis probabilities are already the optimal features.
# Larger MLP + more training + full batch for best accuracy.

n_qubits = 4
lr = 0.003
epochs = 150

data = torch.load('ntangled_states.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(state):
    # Load 4-qubit quantum state and measure computational basis probabilities
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)
    return qml.probs(wires=range(n_qubits))


# Precompute all quantum features (no trainable quantum layers)
print("Computing quantum probability features...")
with torch.no_grad():
    probs_train = torch.stack([circuit(X_train[i]) for i in range(len(X_train))]).float()
    probs_test = torch.stack([circuit(X_test[i]) for i in range(len(X_test))]).float()
print(f"  Train: {probs_train.shape}, Test: {probs_test.shape}")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


model = MLP()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    idx = torch.randperm(len(probs_train))[:500]
    logits = model(probs_train[idx])
    loss = loss_fn(logits, y_train[idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 30 == 0:
        print(f"  epoch {epoch+1:3d}: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(probs_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
