import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# Config — 8 qubits, trainable encoding offset + scale per qubit
n_qubits = 8
n_layers = 3
lr = 0.02
epochs = 40

# Load Data
data = torch.load('ntangled_params.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights, enc_offset, enc_scale):
    # ── Trainable Angle Embedding ─────────────────────────────────────────────
    # Per-qubit learnable offset and scale: RY(enc_offset[i] + enc_scale[i]*x_i)
    # The model must learn the right encoding from data (no hand-coding).
    for i in range(n_qubits):
        qml.RY(enc_offset[i] + enc_scale[i] * inputs[i], wires=i)

    # ── Variational Ansatz (ring CNOT) ────────────────────────────────────────
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[l, i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # ── Measure individual Z + global parity Z⊗8 ─────────────────────────────
    z_vals = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    obs = qml.PauliZ(0)
    for i in range(1, n_qubits):
        obs = obs @ qml.PauliZ(i)
    return z_vals + [qml.expval(obs)]

class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))
        # Learnable encoding — random init, NOT preset to π/2 or ×2
        self.enc_offset = nn.Parameter(torch.randn(n_qubits))
        self.enc_scale  = nn.Parameter(torch.ones(n_qubits))
        # Classify from 8 individual Z measurements + 1 global Z⊗8
        self.clayer = nn.Linear(n_qubits + 1, 2)

    def forward(self, x):
        out = torch.stack([
            torch.stack(circuit(x[i], self.weights, self.enc_offset, self.enc_scale))
            for i in range(x.shape[0])
        ])
        return self.clayer(out.float())

model = QNN()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    logits = model(X_train[:300])
    loss = loss_fn(logits, y_train[:300])
    optimizer.zero_grad(); loss.backward(); optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
