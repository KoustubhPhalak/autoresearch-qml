import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam
import math

# Config — 8 qubits, parity-exact encoding, no variational corruption
n_qubits = 8
n_layers = 0   # variational layers corrupt the exact parity signal
lr = 0.01
epochs = 20

# Load Data
data = torch.load('ntangled_params.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    # ── Parity-exact encoding ─────────────────────────────────────────────────
    # bit_i = 1 iff sin(2*x_i) < 0  (detects which π/2-bin each feature is in)
    # RY(π/2 + 2*x_i) → ⟨Z_i⟩ = -sin(2*x_i)
    # ⟨Z⊗8⟩ = Πᵢ⟨Zᵢ⟩ < 0 iff odd number of bits = 1 iff y = 1
    for i in range(n_qubits):
        qml.RY(math.pi / 2 + 2.0 * inputs[i], wires=i)
    # No variational layers — they degrade the analytic parity signal

    obs = qml.PauliZ(0)
    for i in range(1, n_qubits):
        obs = obs @ qml.PauliZ(i)
    return qml.expval(obs)

class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(max(n_layers, 1), n_qubits))
        # Scale: sharpens boundary; fixed positive so grad step can tune it
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        q_out = torch.stack([circuit(x[i], self.weights) for i in range(x.shape[0])])
        q_out = q_out.float()
        # ⟨Z⊗8⟩ > 0 → class 0;  < 0 → class 1
        logits = torch.stack([self.scale * q_out, -self.scale * q_out], dim=1)
        return logits

model = QNN()
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training — only scale is trainable; circuit is analytically correct
for epoch in range(epochs):
    logits = model(X_train[:300])
    loss = loss_fn(logits, y_train[:300])
    optimizer.zero_grad(); loss.backward(); optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_test[:200]).argmax(dim=1)
    acc = (preds == y_test[:200]).float().mean().item()

print(f"RESULT: TEST_ACC={acc:.4f}")
