import torch
import torch.nn as nn
from torch.optim import Adam
import pennylane as qml
import math
from itertools import combinations

# iter24: On-the-fly feature computation during training.
# Precomputing 2.4M features (iter23) was the bottleneck.
# With on-the-fly computation, we can use 1M+ samples per class.
# The statevectors are much more compact than precomputed features.
# Generate statevectors once; compute 39-dim features per mini-batch during training.

n_classes = 8
N_QUBITS = 4
EXTRA_PER_CLASS = 1000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = torch.load('ntangled_states.pt', weights_only=False)
X_train, y_train = data['X_train'], data['y_train']
X_test,  y_test  = data['X_test'],  data['y_test']
trained_weights = data['trained_weights']

class_keys = [
    "depth_1_ce_005", "depth_1_ce_015", "depth_1_ce_025", "depth_1_ce_035",
    "depth_4_ce_005", "depth_4_ce_015", "depth_4_ce_025", "depth_4_ce_035",
]

dev = qml.device("default.qubit", wires=N_QUBITS)


def get_hwe_matrix(weights):
    @qml.qnode(dev, interface=None, diff_method=None)
    def basis_circ(weights, idx):
        qml.BasisState([int(b) for b in format(idx, f'0{N_QUBITS}b')], wires=range(N_QUBITS))
        depth = weights.shape[0]
        for d in range(depth):
            for q in range(N_QUBITS):
                a, b, c = float(weights[d, q, 0]), float(weights[d, q, 1]), float(weights[d, q, 2])
                qml.Rot(a, b, c, wires=q)
            for q in range(0, N_QUBITS - 1, 2): qml.CZ(wires=[q, q + 1])
            for q in range(N_QUBITS):
                a, b, c = float(weights[d, q, 0]), float(weights[d, q, 1]), float(weights[d, q, 2])
                qml.Rot(a, b, c, wires=q)
            for q in range(1, N_QUBITS - 1, 2): qml.CZ(wires=[q, q + 1])
        return qml.state()

    cols = [torch.tensor(basis_circ(weights, i), dtype=torch.complex64)
            for i in range(2 ** N_QUBITS)]
    return torch.stack(cols, dim=1)


def sample_product_states(n):
    phi = 2 * math.pi * torch.rand(n, N_QUBITS)
    z = 2 * torch.rand(n, N_QUBITS) - 1
    theta = torch.arccos(torch.clamp(z, -1, 1))
    c0 = torch.cos(theta / 2).to(torch.complex64)
    c1 = (torch.sin(theta / 2) * torch.exp(1j * phi)).to(torch.complex64)
    qs = torch.stack([c0, c1], dim=-1)
    states = qs[:, 0, :]
    for q in range(1, N_QUBITS):
        states = torch.einsum('...a,...b->...ab', states, qs[:, q, :]).reshape(n, -1)
    return states


print(f"Generating {EXTRA_PER_CLASS} samples per class...")
extra_X, extra_y = [], []
for cls_id, key in enumerate(class_keys):
    w = trained_weights[key].double()
    U = get_hwe_matrix(w)
    input_states = sample_product_states(EXTRA_PER_CLASS)
    output_states = (U @ input_states.T).T
    extra_X.append(output_states.detach())
    extra_y.append(torch.full((EXTRA_PER_CLASS,), cls_id, dtype=torch.long))
    print(f"  Class {cls_id}: done")

X_train_aug = torch.cat([X_train] + extra_X, dim=0).to(device)  # on GPU
y_train_aug  = torch.cat([y_train] + extra_y, dim=0).to(device)
X_test_gpu   = X_test.to(device)
y_test_gpu   = y_test.to(device)
print(f"Augmented train size: {len(X_train_aug)} on {device}")


def compute_features_batch(psi_batch):
    """Compute 39-dim features on-the-fly for a batch. Input: [B, 16] complex on GPU."""
    probs = psi_batch.abs().pow(2).float()
    total_purity = torch.full((len(psi_batch),), 2.0, device=psi_batch.device)
    pfeat = []
    for r in range(1, 4):
        for s in combinations(range(4), r):
            B = len(psi_batch); n = 4
            complement = [i for i in range(n) if i not in s]
            psi = psi_batch.view(B, 2, 2, 2, 2)
            perm = [0] + [sq + 1 for sq in s] + [c + 1 for c in complement]
            psi_r = psi.permute(*perm).contiguous().view(B, 2 ** r, 2 ** (n - r))
            if r <= n - r:
                rho = torch.bmm(psi_r, psi_r.conj().transpose(1, 2))
            else:
                rho = torch.bmm(psi_r.conj().transpose(1, 2), psi_r)
            p = (rho.abs().pow(2)).sum(dim=[1, 2]).real
            pfeat.append(p)
            total_purity += p
    purities = torch.stack(pfeat, dim=1).float()
    ce = (1.0 - total_purity / 16.0).unsqueeze(1).float()
    B = len(psi_batch); psi = psi_batch.view(B, 2, 2, 2, 2); coh = []
    for q in range(4):
        psi0 = psi.select(q + 1, 0); psi1 = psi.select(q + 1, 1)
        b = (psi0 * psi1.conj()).sum(dim=[1, 2, 3])
        coh.extend([b.real.float(), b.imag.float()])
    coh = torch.stack(coh, dim=1)
    return torch.cat([probs, purities, ce, coh], dim=1)


# Pre-compute test features (small, fast)
with torch.no_grad():
    X_test_f = compute_features_batch(X_test_gpu)


class MLP(nn.Module):
    def __init__(self, hidden=(256, 128, 64), dropouts=(0.12, 0.08, 0.0)):
        super().__init__()
        layers = []
        in_dim = 39
        for h, d in zip(hidden, dropouts):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            if d > 0:
                layers.append(nn.Dropout(d))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


configs = [
    (0.005, 42,  (256, 128, 64), (0.12, 0.08, 0.0)),
    (0.005, 0,   (256, 128, 64), (0.12, 0.08, 0.0)),
    (0.004, 123, (512, 256, 128),(0.10, 0.08, 0.05)),
    (0.006, 7,   (256, 128, 64), (0.10, 0.06, 0.0)),
    (0.005, 99,  (256, 128, 64), (0.12, 0.08, 0.0)),
    (0.003, 13,  (512, 256, 128),(0.12, 0.10, 0.05)),
    (0.007, 256, (256, 128, 64), (0.10, 0.05, 0.0)),
]

all_logits = []
n_train = len(X_train_aug)
loss_fn = nn.CrossEntropyLoss()
steps = 3000

for lr, seed, hidden, dropouts in configs:
    torch.manual_seed(seed)
    model = MLP(hidden, dropouts).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=750, T_mult=2)

    for step in range(steps):
        model.train()
        idx = torch.randperm(n_train, device=device)[:4096]
        batch_psi = X_train_aug[idx]
        with torch.no_grad():
            batch_f = compute_features_batch(batch_psi)
        logits = model(batch_f)
        loss = loss_fn(logits, y_train_aug[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_f).cpu()
        acc = (logits.argmax(dim=1) == y_test).float().mean().item()
    print(f"  lr={lr} seed={seed}: TEST_ACC={acc:.4f}")
    all_logits.append(logits)

ens_logits = torch.stack(all_logits).mean(dim=0)
ens_acc = (ens_logits.argmax(dim=1) == y_test).float().mean().item()
print(f"RESULT: TEST_ACC={ens_acc:.4f}")
