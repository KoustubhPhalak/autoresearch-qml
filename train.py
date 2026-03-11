import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# iter2: StatePrep + trainable basis rotations + all ZZ correlators
# Physics: CE depends on tr(rho_i^2), detectable via ZZ correlators
# No destructive pooling — measure 10 observables then classify classically

n_qubits = 4
lr = 0.03
epochs = 80

data = torch.load('ntangled_states.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(state, u_weights):
    # Load 4-qubit quantum state
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # Trainable basis rotation: find optimal measurement basis
    for i in range(n_qubits):
        qml.RY(u_weights[i, 0], wires=i)
        qml.RZ(u_weights[i, 1], wires=i)

    # Return all single-qubit Z + all pairwise ZZ correlators
    # Single-qubit: 4 values; pairwise: C(4,2)=6 values → 10 total
    obs = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
            for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return obs


class QClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Trainable measurement bases: [n_qubits, 2] (RY angle, RZ angle)
        self.u_weights = nn.Parameter(torch.randn(n_qubits, 2))
        # Classical MLP on 10 quantum features
        self.mlp = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        out = torch.stack([
            torch.stack(circuit(x[i], self.u_weights))
            for i in range(x.shape[0])
        ])
        return self.mlp(out.float())


model = QClassifier()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    idx = torch.randperm(len(X_train))[:300]
    logits = model(X_train[idx])
    loss = loss_fn(logits, y_train[idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
