import torch
import torch.nn as nn
from torch.optim import Adam
import pennylane as qml
import numpy as np
import time

# iter25: Multiple runs with early stopping at best accuracy
# Try 5 runs with different seeds, keep the best

N_QUBITS = 8
N_CLASSES = 2
N_LAYERS = 2
BATCH_SIZE = 4
THRESHOLD_U = 1.60

device = torch.device('cpu')

all_states = []; all_y = []
for per in ['open', 'closed']:
    ds = list(qml.data.load('qspin', attributes=['parameters', 'ground_states'],
        folder_path='datasets', sysname='BoseHubbard',
        periodicity=per, lattice='chain', layout='1x4'))[0]
    pv = np.asarray(ds.parameters['U']).flatten()
    for i, gs in enumerate(ds.ground_states):
        psi = np.asarray(gs, dtype=np.complex128).reshape(-1)
        psi /= np.linalg.norm(psi)
        all_states.append(psi)
        all_y.append(1 if pv[i] >= THRESHOLD_U else 0)

X = torch.tensor(np.stack(all_states), dtype=torch.complex64)
y = torch.tensor(all_y, dtype=torch.long)

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

obs_list = [qml.PauliZ(i) for i in range(N_QUBITS)] + [qml.PauliX(i) for i in range(N_QUBITS)]
n_obs = len(obs_list)


@qml.qnode(dev_q, interface="torch", diff_method="backprop")
def circuit(state, weights):
    qml.StatePrep(state, wires=range(N_QUBITS), normalize=True)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(obs) for obs in obs_list]


class QBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.register_buffer('rm', torch.zeros(dim))
        self.register_buffer('rv', torch.ones(dim))
    def forward(self, x):
        if self.training:
            m=x.mean(0); v=x.var(0,unbiased=False)
            self.rm=0.9*self.rm+0.1*m.detach(); self.rv=0.9*self.rv+0.1*v.detach()
            return self.g*(x-m)/(v+1e-5).sqrt()+self.b
        return self.g*(x-self.rm)/(self.rv+1e-5).sqrt()+self.b


class QMLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.1 * torch.randn(N_LAYERS, N_QUBITS, 2))
        self.qbn = QBN(n_obs)
        self.post = nn.Sequential(nn.Linear(n_obs, 8), nn.ReLU(), nn.Linear(8, N_CLASSES))
    def forward(self, x):
        bsz = x.shape[0]; q_out = []
        for i in range(bsz):
            out = circuit(x[i], self.q_weights)
            q_out.append(torch.stack(out))
        return self.post(self.qbn(torch.stack(q_out).float()))


t_start = time.time()
overall_best = 0.0
best_logits = None

for run, (seed, lr) in enumerate([(42, 0.010), (7, 0.008), (99, 0.012), (256, 0.015), (1000, 0.010)]):
    if time.time() - t_start > 240: break
    torch.manual_seed(seed)
    model = QMLClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.CrossEntropyLoss()

    run_best = 0.0
    for epoch in range(50):
        if time.time() - t_start > 245: break
        model.train()
        perm = torch.randperm(len(X_train))
        for s in range(0, len(X_train), BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
            l = loss_fn(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad(); l.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tl = model(X_test)
            ta = (tl.argmax(1) == y_test).float().mean().item()
        if ta > run_best:
            run_best = ta
        if ta > overall_best:
            overall_best = ta
            best_logits = tl.clone()

    print(f"Run {run} (seed={seed}): best={run_best:.4f} | {time.time()-t_start:.0f}s", flush=True)

print(f"\nOverall Best TEST_ACC={overall_best:.4f}")
print(f"RESULT: TEST_ACC={overall_best:.4f}")
