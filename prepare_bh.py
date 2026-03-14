#!/usr/bin/env python3
"""Prepare combined BoseHubbard open+closed dataset for 3-phase classification."""

import torch
import pennylane as qml
import numpy as np

N_QUBITS = 8  # BoseHubbard 1x4 uses 8-qubit Hilbert space (256 dim)
CLOW, CHIGH = 2.5, 4.5  # Most balanced split
SEED = 1234

rng = np.random.default_rng(SEED)

all_states = []
all_y = []

for periodicity in ['open', 'closed']:
    ds = list(qml.data.load(
        'qspin', attributes=['parameters', 'ground_states'],
        folder_path='datasets', sysname='BoseHubbard',
        periodicity=periodicity, lattice='chain', layout='1x4',
    ))[0]

    states = [np.asarray(gs, dtype=np.complex128).reshape(-1) for gs in ds.ground_states]
    pv = np.asarray(ds.parameters['U']).flatten()
    y = np.where(pv < CLOW, 0, np.where(pv > CHIGH, 2, 1))

    all_states.extend(states)
    all_y.extend(y.tolist())
    print(f"  {periodicity}: {len(states)} samples, classes={[int((y==c).sum()) for c in range(3)]}")

X = np.stack(all_states)  # (200, 256) complex
y = np.array(all_y, dtype=np.int64)
print(f"Combined: {len(X)} samples, classes={[int((y==c).sum()) for c in range(3)]}")

# Stratified 70/30 split
train_idx, test_idx = [], []
for c in range(3):
    ci = rng.permutation(np.where(y == c)[0])
    s = int(0.7 * len(ci))
    train_idx.extend(ci[:s].tolist())
    test_idx.extend(ci[s:].tolist())

train_idx = rng.permutation(train_idx)
test_idx = rng.permutation(test_idx)

# Save complex statevectors + labels
payload = {
    'X_train': torch.tensor(X[train_idx], dtype=torch.complex64),
    'y_train': torch.tensor(y[train_idx], dtype=torch.long),
    'X_test': torch.tensor(X[test_idx], dtype=torch.complex64),
    'y_test': torch.tensor(y[test_idx], dtype=torch.long),
    'n_qubits': N_QUBITS,
    'n_classes': 3,
    'label_names': ['superfluid-like', 'critical-like', 'mott-like'],
    'thresholds': (CLOW, CHIGH),
}

torch.save(payload, 'bosehubbard_data.pt')
print(f"Saved bosehubbard_data.pt")
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
