import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# Config
n_qubits = 4
n_layers = 3
lr = 0.02
epochs = 25

# Load Data (NTangled Parameters)
data = torch.load('ntangled_params.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    # ── AGENT: Experiment with Angle Embedding Patterns ──
    # Current: RX on all, then RY on all (2 features per qubit)
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i + n_qubits], wires=i)

    # ── Variational Ansatz ──
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[l, i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))
        self.clayer = nn.Linear(n_qubits, 2)

    def forward(self, x):
        q_out = torch.stack([torch.tensor(circuit(x[i], self.weights)) for i in range(x.shape[0])])
        return self.clayer(q_out.float())

model = QNN()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training loop (Subset for speed)
for epoch in range(epochs):
    logits = model(X_train[:200])
    loss = loss_fn(logits, y_train[:200])
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test[:100]).argmax(dim=1)
    acc = (preds == y_test[:100]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")