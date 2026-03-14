import torch
import torch.nn as nn
from torch.optim import Adam
import pennylane as qml
import numpy as np
import time

# iter20: Open boundary only — 100% achievable
# StatePrep → 1 variational layer → PauliZ → QBN → 2 linear layers

N_QUBITS = 8
N_CLASSES = 2
N_LAYERS = 1
LR = 0.01
BATCH_SIZE = 4
THRESHOLD_U = 1.60

device = torch.device('cpu')

ds = list(qml.data.load('qspin', attributes=['parameters', 'ground_states'],
    folder_path='datasets', sysname='BoseHubbard',
    periodicity='open', lattice='chain', layout='1x4'))[0]
pv = np.asarray(ds.parameters['U']).flatten()

states = []
labels = []
for i, gs in enumerate(ds.ground_states):
    psi = np.asarray(gs, dtype=np.complex128).reshape(-1)
    psi /= np.linalg.norm(psi)
    states.append(psi)
    labels.append(1 if pv[i] >= THRESHOLD_U else 0)

X = torch.tensor(np.stack(states), dtype=torch.complex64)
y = torch.tensor(labels, dtype=torch.long)
print(f"Total: {len(X)} (SF:{(y==0).sum()}, Ins:{(y==1).sum()})")

rng = np.random.default_rng(1234)
train_idx, test_idx = [], []
for c in range(2):
    ci = rng.permutation(np.where(y.numpy() == c)[0])
    s = int(0.7 * len(ci))
    train_idx.extend(ci[:s].tolist())
    test_idx.extend(ci[s:].tolist())

X_train = X[train_idx] / X[train_idx].norm(dim=1, keepdim=True)
y_train = y[train_idx]
X_test = X[test_idx] / X[test_idx].norm(dim=1, keepdim=True)
y_test = y[test_idx]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

dev_q = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_q, interface="torch", diff_method="backprop")
def circuit(state, weights):
    qml.StatePrep(state, wires=range(N_QUBITS), normalize=True)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumBatchNorm(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.momentum = momentum; self.eps = eps
    def forward(self, x):
        if self.training:
            m = x.mean(0); v = x.var(0, unbiased=False)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*m.detach()
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*v.detach()
            x = (x-m)/(v+self.eps).sqrt()
        else:
            x = (x-self.running_mean)/(self.running_var+self.eps).sqrt()
        return self.gamma*x+self.beta


class QMLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.1 * torch.randn(N_LAYERS, N_QUBITS, 2))
        self.qbn = QuantumBatchNorm(N_QUBITS)
        self.post = nn.Sequential(nn.Linear(N_QUBITS, 8), nn.ReLU(), nn.Linear(8, N_CLASSES))
    def forward(self, x):
        bsz = x.shape[0]; q_out = []
        for i in range(bsz):
            out = circuit(x[i], self.q_weights)
            q_out.append(torch.stack(out))
        q_out = torch.stack(q_out).float()
        return self.post(self.qbn(q_out))


model = QMLClassifier().to(device)
optimizer = Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0.0
t_start = time.time()

for epoch in range(80):
    if time.time() - t_start > 250: break
    model.train()
    perm = torch.randperm(len(X_train))
    epoch_loss = 0.0; nb = 0
    for s in range(0, len(X_train), BATCH_SIZE):
        idx = perm[s:s+BATCH_SIZE]
        logits = model(X_train[idx])
        loss = loss_fn(logits, y_train[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_loss += loss.item(); nb += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            ta = (model(X_test).argmax(1) == y_test).float().mean().item()
        if ta > best_acc: best_acc = ta
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/nb:.4f} | Test: {ta:.4f}", flush=True)

model.eval()
with torch.no_grad():
    fa = (model(X_test).argmax(1) == y_test).float().mean().item()

print(f"\nBest TEST_ACC={best_acc:.4f}")
print(f"RESULT: TEST_ACC={fa:.4f}")
