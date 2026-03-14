import torch
import torch.nn as nn
from torch.optim import Adam
import pennylane as qml
import numpy as np
import math

# iter0: Baseline — zx_probs features, angle embed into 4-qubit circuit
# BoseHubbard 3-phase: superfluid / critical / mott
# Use smaller batches and fewer epochs to fit in 300s

N_QUBITS = 4
N_CLASSES = 3
N_LAYERS = 2
N_EPOCHS = 60
LR = 0.02
BATCH_SIZE = 8

device = torch.device('cpu')

data = torch.load('bosehubbard_data.pt', weights_only=False)
X_train_raw = data['X_train'].numpy()
y_train = data['y_train']
X_test_raw = data['X_test'].numpy()
y_test = data['y_test']


def extract_features(states):
    """Z-basis probabilities only (256 features) — skip X-basis to halve feature dim."""
    feats = []
    for s in states:
        s = s.reshape(-1).astype(np.complex128)
        z_probs = np.abs(s) ** 2
        z_probs /= z_probs.sum()
        feats.append(z_probs.astype(np.float32))
    return torch.tensor(np.stack(feats))


print("Extracting features...")
X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)
feat_dim = X_train.shape[1]
print(f"Feature dim: {feat_dim}, Train: {len(X_train)}, Test: {len(X_test)}")

mean = X_train.mean(dim=0)
std = X_train.std(dim=0).clamp(min=1e-6)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

dev_q = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_q, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QMLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(feat_dim, N_QUBITS),
        )
        self.q_weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 2))
        self.post = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        bsz = x.shape[0]
        x = self.pre(x)
        q_out = []
        for i in range(bsz):
            out = circuit(x[i], self.q_weights)
            q_out.append(torch.stack(out))
        q_out = torch.stack(q_out).float()
        return self.post(q_out)


model = QMLClassifier().to(device)
optimizer = Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

X_train_d, y_train_d = X_train.to(device), y_train.to(device)
X_test_d, y_test_d = X_test.to(device), y_test.to(device)

best_acc = 0.0

for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_d))
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, len(X_train_d), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        logits = model(X_train_d[idx])
        loss = loss_fn(logits, y_train_d[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_d)
            test_acc = (test_logits.argmax(1) == y_test_d).float().mean().item()
        if test_acc > best_acc:
            best_acc = test_acc
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | Test: {test_acc:.4f}", flush=True)

model.eval()
with torch.no_grad():
    test_logits = model(X_test_d)
    final_acc = (test_logits.argmax(1) == y_test_d).float().mean().item()

print(f"\nBest TEST_ACC={best_acc:.4f}")
print(f"RESULT: TEST_ACC={final_acc:.4f}")
