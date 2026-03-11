import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# iter9: True hybrid — StatePrep + variational layers (zero init) + probs + MLP
# Lessons from iter4 (worked, 0.945): ZERO init, lr=0.005, deep MLP head.
# Lessons from iter8 (failed, 0.455): 0.01*randn init + lr=0.05 + Linear → plateau.
# Fix: zero quantum weight init + lr=0.003 + BN+MLP head to extract from probs.

n_qubits = 4
n_layers = 2
lr = 0.003
epochs = 80

data = torch.load('ntangled_states.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(state, weights):
    """
    weights: (n_layers, n_qubits, 2) trainable RY/RZ angles.
    Zero-initialized → starts as near-identity; gradients flow via torch backprop.
    """
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    return qml.probs(wires=range(n_qubits))


class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Zero init — starts at identity, gradient non-vanishing near zero
        self.q_weights = nn.Parameter(torch.zeros(n_layers, n_qubits, 2))
        self.mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out = torch.stack([
            circuit(x[i], self.q_weights) for i in range(x.shape[0])
        ])
        return self.mlp(out.float())


model = QNN()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    idx = torch.randperm(len(X_train))[:300]
    logits = model(X_train[idx])
    loss = loss_fn(logits, y_train[idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 10 == 0:
        # Print gradient magnitude for quantum weights as a health check
        q_grad = model.q_weights.grad
        gn = q_grad.norm().item() if q_grad is not None else 0.0
        print(f"  epoch {epoch+1:3d}: loss={loss.item():.4f}  |q_grad|={gn:.5f}")

model.eval()
with torch.no_grad():
    preds = model(X_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
