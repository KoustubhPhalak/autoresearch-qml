"""
NTangled-style quantum dataset generator
=========================================
Follows the methodology of Schatzki et al. (2021) — arXiv:2109.03400

Pipeline:
  1. Sample Haar-random product-state inputs
  2. Train a parameterized quantum circuit (PQC) so output states
     match a target concentratable entanglement (CE) value
  3. Materialize a dataset of statevectors + labels from trained generators

CE definition follows Beckey et al. (arXiv:2104.06923), Definition 1:
  CE(|ψ⟩) = 1 - (1/2^n) * Σ_α Tr(ρ_α²)
No extra normalisation prefactor.  CE(|GHZ_n⟩) = 1/2 - 1/2^n.

Notes:
  - Uses a hardware-efficient brick-layer ansatz with even/odd CZ
    entangling pairs.  Same rotation parameters are applied to both
    sublayers within each depth block, matching the repo pseudocode.
  - Product-state inputs are Haar-random on each qubit's Bloch sphere.
  - This is the trained-generator NTangled dataset, NOT the separate
    depth-learning benchmark from Section V of the paper.
"""

import math
import itertools
import numpy as np
import torch
import pennylane as qml
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
N_QUBITS = 4
ANSATZ_DEPTHS = [1, 4]
TARGET_CES = [0.05, 0.15, 0.25, 0.35]  # matches released 4-qubit HWE target values
SAMPLES_PER_GENERATOR = 200

TRAIN_STEPS = 300
BATCH_SIZE = 16
LR = 5e-2
SEED = 1234

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

dev = qml.device("default.qubit", wires=N_QUBITS)


# ─────────────────────────────────────────────
# Concentratable entanglement (exact, pure state)
# ─────────────────────────────────────────────
def concentratable_entanglement(state, n_qubits=N_QUBITS):
    """
    Exact CE for an n-qubit pure statevector (Beckey et al., Def. 1):

        CE(|ψ⟩) = 1 - (1/2^n) * Σ_α Tr(ρ_α²)

    where the sum runs over all 2^n subsets α ⊆ {0, …, n-1}.

    No extra normalisation prefactor — this matches Beckey's definition
    directly. For reference, CE(|GHZ_n⟩) = 1/2 - 1/2^n (= 7/16 at n=4).

    For the empty set and full system, Tr(ρ²) = 1 (pure state),
    so we add those analytically and skip the density-matrix work.
    """
    dim = 2 ** n_qubits

    # Empty set (r=0) and full system (r=n) each contribute purity 1
    total = state.real.new_tensor(2.0)

    for r in range(1, n_qubits):
        for subset in itertools.combinations(range(n_qubits), r):
            rho = qml.math.reduce_statevector(state, indices=list(subset))
            purity = qml.math.real(qml.math.trace(rho @ rho))
            total = total + purity

    return 1.0 - total / dim


# ─────────────────────────────────────────────
# Random product-state input distribution
# ─────────────────────────────────────────────
def sample_product_state_angles(batch_size, n_qubits=N_QUBITS):
    """
    Haar-random single-qubit states for each qubit.

    Parameterisation: |ψ⟩ = RZ(φ) RY(θ) |0⟩
      φ  ~ Uniform[0, 2π)
      θ  = arccos(z),  z ~ Uniform[-1, 1]

    Returns: Tensor [batch, n_qubits, 2]  (last dim = [φ, θ])
    """
    phi = 2.0 * math.pi * torch.rand(batch_size, n_qubits)
    z = 2.0 * torch.rand(batch_size, n_qubits) - 1.0
    theta = torch.arccos(torch.clamp(z, -1.0, 1.0))
    return torch.stack([phi, theta], dim=-1)


# ─────────────────────────────────────────────
# Hardware-efficient ansatz (even/odd brick-layer CZ)
# ─────────────────────────────────────────────
def hwe_ansatz(weights, n_qubits=N_QUBITS):
    """
    Hardware-efficient brick-layer ansatz matching the NTangled repo
    generator pseudocode:
      Per depth block d:
        1. U3(θ_{d,q}) on every qubit q
        2. CZ on even pairs: (0,1), (2,3), …
        3. U3(θ_{d,q}) on every qubit q   (same parameters reused)
        4. CZ on odd pairs:  (1,2), (3,4), …

    weights shape: [depth, n_qubits, 3]
      The repo pseudocode shows a single θ_{d,i,0:3} tensor per depth
      layer, applied to both rotation sublayers within that block.
    """
    depth = weights.shape[0]
    for d in range(depth):
        for q in range(n_qubits):
            a, b, c = weights[d, q]
            qml.Rot(a, b, c, wires=q)
        for q in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
        for q in range(n_qubits):
            a, b, c = weights[d, q]
            qml.Rot(a, b, c, wires=q)
        for q in range(1, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])


@qml.qnode(dev, interface="torch", diff_method="backprop")
def generator_circuit(weights, input_angles):
    """
    Full generator: random product-state input → HWE ansatz → output state.
    """
    n_qubits = weights.shape[1]
    for q in range(n_qubits):
        phi, theta = input_angles[q]
        qml.RY(theta, wires=q)
        qml.RZ(phi, wires=q)
    hwe_ansatz(weights, n_qubits=n_qubits)
    return qml.state()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def batch_ce_loss(weights, batch_inputs, target_ce):
    """MSE between realised CE and target CE over a batch of product-state inputs."""
    target = torch.as_tensor(target_ce, dtype=weights.dtype)
    losses = []
    for x in batch_inputs:
        state = generator_circuit(weights, x)
        ce = concentratable_entanglement(state)
        losses.append((ce - target) ** 2)
    return torch.mean(torch.stack(losses))


def train_generator(
    target_ce,
    depth,
    n_qubits=N_QUBITS,
    steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    lr=LR,
):
    """
    Train a single HWE generator to produce states with a given CE.
    Returns the best weights found during training.
    """
    weights = (2.0 * math.pi * torch.rand(depth, n_qubits, 3)).requires_grad_()
    opt = torch.optim.Adam([weights], lr=lr)

    best_weights = None
    best_loss = float("inf")

    for step in range(steps):
        batch_inputs = sample_product_state_angles(batch_size, n_qubits=n_qubits)
        loss = batch_ce_loss(weights, batch_inputs, target_ce)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = weights.detach().clone()

        if step % 50 == 0 or step == steps - 1:
            print(
                f"  [depth={depth}  target_ce={target_ce:.2f}]  "
                f"step {step:03d}  loss={loss_val:.6f}"
            )

    print(f"  Best loss: {best_loss:.6f}")
    return best_weights


# ─────────────────────────────────────────────
# Dataset materialisation
# ─────────────────────────────────────────────
def materialize_states(trained_weights, n_states, n_qubits=N_QUBITS):
    """
    Feed fresh random product-state inputs through a trained generator
    and record the output statevectors and their realised CE values.
    """
    states = []
    ces = []

    for _ in range(n_states):
        x = sample_product_state_angles(1, n_qubits=n_qubits)[0]
        with torch.no_grad():
            state = generator_circuit(trained_weights, x)
            ce = concentratable_entanglement(state, n_qubits=n_qubits)

        states.append(state.cpu().numpy())
        ces.append(float(ce.cpu()))

    return np.asarray(states), np.asarray(ces)


# ─────────────────────────────────────────────
# Full dataset builder
# ─────────────────────────────────────────────
def build_ntangled_dataset(
    depths=ANSATZ_DEPTHS,
    target_ces=TARGET_CES,
    samples_per_generator=SAMPLES_PER_GENERATOR,
    n_qubits=N_QUBITS,
    test_size=0.2,
    save_path="ntangled_states.pt",
):
    """
    Train one generator per (depth, target_ce) pair, materialise states,
    and package everything into a single .pt file.

    Label convention:
        class_id = sequential integer per (depth, target_ce) combination.

    Saved metadata includes target CE, generator depth, and realised CE
    per sample so downstream tasks can define their own label splits
    (e.g. binary low/high CE, multi-class by CE bucket, regression, etc.).
    """
    all_X, all_y = [], []
    all_target_ce, all_depth, all_realized_ce = [], [], []
    trained_weights = {}

    class_id = 0
    for depth in depths:
        for tce in target_ces:
            print(f"\n{'='*60}")
            print(f"Training generator:  depth={depth}  target_ce={tce:.2f}")
            print(f"{'='*60}")

            w = train_generator(target_ce=tce, depth=depth, n_qubits=n_qubits)
            key = f"depth_{depth}_ce_{int(round(tce * 100)):03d}"
            trained_weights[key] = w.cpu()

            states, ces = materialize_states(w, samples_per_generator, n_qubits=n_qubits)

            all_X.append(states)
            all_y.append(np.full(samples_per_generator, class_id, dtype=np.int64))
            all_target_ce.append(np.full(samples_per_generator, tce))
            all_depth.append(np.full(samples_per_generator, depth, dtype=np.int64))
            all_realized_ce.append(ces)

            mean_ce = ces.mean()
            std_ce = ces.std()
            print(f"  Realised CE:  {mean_ce:.4f} ± {std_ce:.4f}  (target {tce:.2f})")

            class_id += 1

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    target_ce_arr = np.concatenate(all_target_ce)
    depth_arr = np.concatenate(all_depth)
    realized_ce_arr = np.concatenate(all_realized_ce)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=SEED, stratify=y,
    )

    payload = {
        # core data
        "X_train": torch.tensor(X[train_idx], dtype=torch.complex64),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "X_test": torch.tensor(X[test_idx], dtype=torch.complex64),
        "y_test": torch.tensor(y[test_idx], dtype=torch.long),
        # metadata
        "target_ce_train": torch.tensor(target_ce_arr[train_idx], dtype=torch.float32),
        "target_ce_test": torch.tensor(target_ce_arr[test_idx], dtype=torch.float32),
        "depth_train": torch.tensor(depth_arr[train_idx], dtype=torch.long),
        "depth_test": torch.tensor(depth_arr[test_idx], dtype=torch.long),
        "realized_ce_train": torch.tensor(realized_ce_arr[train_idx], dtype=torch.float32),
        "realized_ce_test": torch.tensor(realized_ce_arr[test_idx], dtype=torch.float32),
        # generators (so you can materialise more states later)
        "trained_weights": trained_weights,
        # config for reproducibility
        "config": {
            "n_qubits": n_qubits,
            "depths": depths,
            "target_ces": target_ces,
            "samples_per_generator": samples_per_generator,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
        },
    }

    torch.save(payload, save_path)

    n_classes = class_id
    n_total = len(X)
    print(f"\n{'='*60}")
    print(f"Dataset saved to {save_path}")
    print(f"  Total states : {n_total}")
    print(f"  Classes      : {n_classes}")
    print(f"  Train / Test : {len(train_idx)} / {len(test_idx)}")
    print(f"{'='*60}")

    return payload


# ─────────────────────────────────────────────
if __name__ == "__main__":
    build_ntangled_dataset()