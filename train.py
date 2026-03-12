import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# iter37: QBN + wide Linear(16,256) + ReLU + Linear(256,2). No classical BN.
#
# Permanent constraints: QBN on QNN output, max 2 linear layers, no classical BN, 100 epochs.
# User clarified: width is unconstrained, just max 2 linear layers.
#
# Root cause of iter33 (0.935) plateau:
#   Linear(16,64) → |q_grad| ≈ 0.014–0.020. Not enough for deeper convergence.
#   In iter9, BN(64) after Linear(16,64) boosted |q_grad| to 0.017–0.026.
#   Without BN inside head, the gradient flowing back to quantum circuit is weaker.
#
# Fix: Use Linear(16,256) — the 256 units create a much larger gradient projection
#   from the first layer back to the 16-dim probs.
#   ∂L/∂probs ∝ W^T (256×16 matrix) → 16 stronger gradient signals.
#   Expected |q_grad| ≈ 0.025–0.045, closer to iter9's range.
#
# lr=0.006, batch=300, cosine annealing.

n_qubits = 4
n_layers = 2
lr = 0.006
epochs = 100

data = torch.load('ntangled_states.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def circuit(state, weights):
    """Pure QNN: StatePrep + 2-layer CNOT-ring. No BN anywhere inside."""
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
        self.q_weights = nn.Parameter(torch.zeros(n_layers, n_qubits, 2))
        # QBN: only normalisation.
        self.q_bn = nn.BatchNorm1d(16)
        # Wide head: 256 units amplify gradient back to quantum circuit.
        self.head = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        probs = torch.stack([
            circuit(x[i], self.q_weights) for i in range(x.shape[0])
        ]).float()
        return self.head(self.q_bn(probs))


model = QNN()
optimizer = Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    idx = torch.randperm(len(X_train))[:300]
    logits = model(X_train[idx])
    loss = loss_fn(logits, y_train[idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()
    if (epoch + 1) % 25 == 0:
        q_grad = model.q_weights.grad
        gn = q_grad.norm().item() if q_grad is not None else 0.0
        print(f"  epoch {epoch+1:3d}: loss={loss.item():.4f}  |q_grad|={gn:.5f}")

model.eval()
with torch.no_grad():
    preds = model(X_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
