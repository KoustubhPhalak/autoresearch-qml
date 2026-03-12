import torch
import torch.nn as nn
import pennylane as qml
from torch.optim import Adam

# iter38: TRUE QBN inside circuit + wide Linear(16,256) + ReLU + Linear(256,2).
#
# iter37 used nn.BatchNorm1d(16) which is classical BN applied to quantum output (CBN).
# This is NOT QBN. True QBN applies batch-stat corrections as RY gates INSIDE the circuit.
#
# True QBN (permanent):
#   1. Compute raw probs directly from input statevectors: p_raw = |x|^2 (no extra circuit pass).
#   2. Batch-mean probs p_bar → per-qubit marginals m_i = P(qubit_i = |0>).
#   3. Correction angle: θ_i = γ_i * arcsin(1 - 2*m_i) + β_i
#      (γ_i, β_i are learned BN scale/shift parameters, 4 each).
#   4. Apply RY(θ_i) on qubit i inside the circuit, BEFORE variational layers.
#
# This is quantum: the normalisation is done in Hilbert space via unitary rotations,
# not by arithmetically rescaling floating-point activations.
#
# Head unchanged: Linear(16,256) → ReLU → Linear(256,2).
# lr=0.006, batch=300, cosine annealing, 100 epochs.

n_qubits = 4
n_layers = 2
lr = 0.006
epochs = 100

data = torch.load('ntangled_states.pt')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

dev = qml.device("default.qubit", wires=n_qubits)

# Precompute masks: masks[i] is a bool tensor of length 16 where True = qubit i is |0>.
# PennyLane orders state k as: k = b0*2^(n-1) + b1*2^(n-2) + ... + b_{n-1}*1
# So qubit i is 0 iff bit (n-1-i) of k is 0.
_masks = torch.stack([
    torch.tensor([(k >> (n_qubits - 1 - i)) & 1 == 0 for k in range(2**n_qubits)], dtype=torch.float32)
    for i in range(n_qubits)
])  # shape (4, 16)


@qml.qnode(dev, interface="torch")
def circuit(state, weights, theta_corr):
    """True hybrid QNN with QBN: StatePrep + QBN correction rotations + variational layers."""
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)
    # QBN: per-qubit correction rotation applied inside the circuit
    for i in range(n_qubits):
        qml.RY(theta_corr[i], wires=i)
    # Variational layers
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
        # QBN learnable parameters: γ (scale) and β (shift) per qubit
        self.qbn_gamma = nn.Parameter(torch.ones(n_qubits))
        self.qbn_beta = nn.Parameter(torch.zeros(n_qubits))
        # Wide head: 256 units amplify gradient back to quantum circuit
        self.head = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        # QBN: compute batch statistics from raw statevectors (|amplitude|^2, no circuit needed)
        probs_raw = x.abs().pow(2).float()        # (batch, 16)
        p_bar = probs_raw.mean(0).detach()        # (16,), treated as constant like classical BN

        # Per-qubit marginals: m_i = P(qubit_i = |0>)
        marginals = (_masks * p_bar.unsqueeze(0)).sum(1)  # (4,)

        # Correction angles with learned γ, β (gradients flow through these)
        theta_corr = self.qbn_gamma * torch.arcsin(
            torch.clamp(1 - 2 * marginals, -1 + 1e-6, 1 - 1e-6)
        ) + self.qbn_beta

        # Run circuit with QBN correction applied inside
        probs = torch.stack([
            circuit(x[i], self.q_weights, theta_corr) for i in range(x.shape[0])
        ]).float()

        return self.head(probs)


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
