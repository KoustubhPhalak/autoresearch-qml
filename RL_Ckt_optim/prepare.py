"""
Quantum Circuit Compilation — Benchmark & Baseline Preparation
===============================================================
Runs ONCE before the RL agent loop.

Generates:
  1. Random Haar unitaries (100 each for 3, 4, 5 qubits)
  2. QFT unitaries (3–5 qubits)
  3. Grover oracle unitaries (3–5 qubits, multiple marked states)
  4. Hamiltonian simulation unitaries (Heisenberg & transverse-field Ising)

Computes baselines:
  - Qiskit transpiler gate counts at optimization levels 0–3
    (IBM-native: CX + U3;  Google-native: CZ + single-qubit)
  - KAK decomposition gate counts for 2-qubit unitary sub-blocks

Creates a FIXED train / val / test split (60 / 20 / 20), stratified by
(category, n_qubits).  The split is deterministic (seeded) and saved
inside benchmark_data.pt so every training run uses the same partition.

Saves everything to  benchmark_data.pt  for the RL agent.
"""

import time
import math
import itertools
import numpy as np
from collections import defaultdict
from scipy.stats import unitary_group

import torch
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import CXGate

# TwoQubitBasisDecomposer moved from qiskit.quantum_info to qiskit.synthesis in Qiskit 1.0
try:
    from qiskit.synthesis import TwoQubitBasisDecomposer
except ImportError:
    from qiskit.quantum_info import TwoQubitBasisDecomposer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEED = 42
N_HAAR_SAMPLES = 100          # per qubit count
QUBIT_RANGE = [3, 4, 5]
GROVER_MARKED_STATES = 3      # number of distinct marked-state oracles per qubit count
TROTTER_STEPS = [1, 2, 4]     # for Hamiltonian simulation circuits
QISKIT_OPT_LEVELS = [0, 1, 2, 3]

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Unitary Generators
# ─────────────────────────────────────────────

def generate_haar_unitaries(n_qubits: int, n_samples: int) -> list[np.ndarray]:
    """Sample Haar-random unitaries from the circular unitary ensemble."""
    dim = 2 ** n_qubits
    return [unitary_group.rvs(dim) for _ in range(n_samples)]


def generate_qft_unitary(n_qubits: int) -> np.ndarray:
    """Return the exact QFT unitary matrix for n qubits."""
    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    rows = np.arange(N)
    U = omega ** np.outer(rows, rows) / np.sqrt(N)
    return U


def generate_grover_oracle(n_qubits: int, marked_state: int) -> np.ndarray:
    """Grover oracle: U = I - 2|m⟩⟨m|"""
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=complex)
    U[marked_state, marked_state] = -1.0
    return U


def generate_grover_diffusion(n_qubits: int) -> np.ndarray:
    """Grover diffusion: 2|s⟩⟨s| - I  where |s⟩ = H^{⊗n}|0⟩."""
    dim = 2 ** n_qubits
    s = np.ones(dim, dtype=complex) / np.sqrt(dim)
    return 2.0 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)


def generate_grover_iterate(n_qubits: int, marked_state: int) -> np.ndarray:
    """Single Grover iterate G = D · O."""
    O = generate_grover_oracle(n_qubits, marked_state)
    D = generate_grover_diffusion(n_qubits)
    return D @ O


def heisenberg_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """1D Heisenberg XXX chain with longitudinal field."""
    dim = 2 ** n_qubits
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_op(op, qubit, nq):
        ops = [I] * nq; ops[qubit] = op
        result = ops[0]
        for o in ops[1:]: result = np.kron(result, o)
        return result

    def kron_two(op1, q1, op2, q2, nq):
        ops = [I] * nq; ops[q1] = op1; ops[q2] = op2
        result = ops[0]
        for o in ops[1:]: result = np.kron(result, o)
        return result

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        H += J * (kron_two(X, i, X, i+1, n_qubits) +
                  kron_two(Y, i, Y, i+1, n_qubits) +
                  kron_two(Z, i, Z, i+1, n_qubits))
    for i in range(n_qubits):
        H += h * kron_op(Z, i, n_qubits)
    return H


def tfim_hamiltonian(n_qubits: int, J: float = 1.0, g: float = 1.0) -> np.ndarray:
    """Transverse-field Ising model (1D, open boundary)."""
    dim = 2 ** n_qubits
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_op(op, qubit, nq):
        ops = [I] * nq; ops[qubit] = op
        result = ops[0]
        for o in ops[1:]: result = np.kron(result, o)
        return result

    def kron_two(op1, q1, op2, q2, nq):
        ops = [I] * nq; ops[q1] = op1; ops[q2] = op2
        result = ops[0]
        for o in ops[1:]: result = np.kron(result, o)
        return result

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        H -= J * kron_two(Z, i, Z, i+1, n_qubits)
    for i in range(n_qubits):
        H -= g * kron_op(X, i, n_qubits)
    return H


def trotter_unitary(hamiltonian: np.ndarray, t: float, steps: int) -> np.ndarray:
    """First-order Trotter: U(t) ≈ (e^{-iHt/n})^n"""
    from scipy.linalg import expm
    dt = t / steps
    U_step = expm(-1j * hamiltonian * dt)
    U = np.eye(hamiltonian.shape[0], dtype=complex)
    for _ in range(steps):
        U = U_step @ U
    return U


# ─────────────────────────────────────────────
# Qiskit Baseline Compilation
# ─────────────────────────────────────────────

def unitary_to_circuit(U: np.ndarray, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U, list(range(n_qubits)))
    return qc


def count_gates(transpiled_circuit: QuantumCircuit) -> dict:
    ops = transpiled_circuit.count_ops()
    two_q_gates = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "swap", "iswap"}
    n_two_qubit = sum(v for k, v in ops.items() if k in two_q_gates)
    n_total = sum(ops.values())
    return {
        "total_gates": n_total,
        "two_qubit_gates": n_two_qubit,
        "depth": transpiled_circuit.depth(),
        "gate_counts": dict(ops),
    }


def compile_baselines_qiskit(U, n_qubits, label, opt_levels=QISKIT_OPT_LEVELS):
    results = {}
    basis_sets = {
        "ibm": ["cx", "id", "rz", "sx", "x"],
        "google": ["cz", "id", "rx", "ry", "rz"],
    }
    qc = unitary_to_circuit(U, n_qubits)

    for hw_name, basis_gates in basis_sets.items():
        for opt_level in opt_levels:
            key = f"{hw_name}_opt{opt_level}"
            t0 = time.perf_counter()
            try:
                transpiled = transpile(
                    qc, basis_gates=basis_gates,
                    optimization_level=opt_level, seed_transpiler=SEED,
                )
                wall_time = time.perf_counter() - t0
                gate_info = count_gates(transpiled)
                results[key] = {**gate_info, "wall_time_s": wall_time, "success": True}
            except Exception as e:
                wall_time = time.perf_counter() - t0
                results[key] = {
                    "total_gates": -1, "two_qubit_gates": -1, "depth": -1,
                    "gate_counts": {}, "wall_time_s": wall_time,
                    "success": False, "error": str(e),
                }
            print(
                f"    {label:30s}  {key:16s}  "
                f"total={results[key]['total_gates']:4d}  "
                f"2q={results[key]['two_qubit_gates']:3d}  "
                f"time={results[key]['wall_time_s']:.3f}s"
            )
    return results


# ─────────────────────────────────────────────
# KAK Decomposition Baseline (2-qubit sub-blocks)
# ─────────────────────────────────────────────

def kak_baseline_two_qubit(U_4x4):
    try:
        decomposer = TwoQubitBasisDecomposer(
            gate=CXGate(), euler_basis="ZSX",
        )
        qc = decomposer(Operator(U_4x4).data)
        return {**count_gates(qc), "success": True}
    except Exception as e:
        return {
            "total_gates": -1, "two_qubit_gates": -1, "depth": -1,
            "gate_counts": {}, "success": False, "error": str(e),
        }


# ─────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────

def create_stratified_split(benchmarks, seed=SEED):
    """
    Create a deterministic, stratified train/val/test split.

    Stratification key: (category, n_qubits).
    Within each stratum, labels are shuffled with `seed` and split
    60/20/20. This ensures every (category, qubit-count) group is
    represented in all three partitions.

    Returns: train_keys, val_keys, test_keys (lists of benchmark label strings).
    """
    rng = np.random.default_rng(seed)

    # Group labels by stratification key
    strata = defaultdict(list)
    for label, info in benchmarks.items():
        key = (info["category"], info["n_qubits"])
        strata[key].append(label)

    train_keys, val_keys, test_keys = [], [], []

    for stratum_key in sorted(strata.keys()):
        labels = sorted(strata[stratum_key])  # sort first for determinism
        rng.shuffle(labels)
        n = len(labels)
        n_train = max(1, int(round(TRAIN_RATIO * n)))
        n_val   = max(1, int(round(VAL_RATIO * n)))
        # test gets the remainder (avoids off-by-one rounding)
        train_keys.extend(labels[:n_train])
        val_keys.extend(labels[n_train:n_train + n_val])
        test_keys.extend(labels[n_train + n_val:])

    # Verify no overlap
    all_keys = set(train_keys) | set(val_keys) | set(test_keys)
    assert len(all_keys) == len(train_keys) + len(val_keys) + len(test_keys), \
        "Split has overlapping keys!"
    assert all_keys == set(benchmarks.keys()), \
        "Split does not cover all benchmarks!"

    return train_keys, val_keys, test_keys


# ─────────────────────────────────────────────
# Main Dataset Builder
# ─────────────────────────────────────────────

def build_benchmark_dataset(save_path: str = "benchmark_data.pt"):
    benchmarks = {}

    # ── 1. Random Haar Unitaries ──
    print("\n" + "=" * 70)
    print("GENERATING RANDOM HAAR UNITARIES")
    print("=" * 70)
    for nq in QUBIT_RANGE:
        print(f"\n  n_qubits = {nq}  ({N_HAAR_SAMPLES} samples)")
        unitaries = generate_haar_unitaries(nq, N_HAAR_SAMPLES)
        for i, U in enumerate(unitaries):
            label = f"haar_q{nq}_{i:03d}"
            baselines = compile_baselines_qiskit(U, nq, label)
            benchmarks[label] = {
                "unitary": U, "n_qubits": nq,
                "category": "haar", "baselines": baselines,
            }

    # ── 2. QFT Unitaries ──
    print("\n" + "=" * 70)
    print("GENERATING QFT UNITARIES")
    print("=" * 70)
    for nq in QUBIT_RANGE:
        label = f"qft_q{nq}"
        print(f"\n  {label}")
        U = generate_qft_unitary(nq)
        baselines = compile_baselines_qiskit(U, nq, label)
        benchmarks[label] = {
            "unitary": U, "n_qubits": nq,
            "category": "qft", "baselines": baselines,
        }

    # ── 3. Grover Oracles & Iterates ──
    print("\n" + "=" * 70)
    print("GENERATING GROVER CIRCUITS")
    print("=" * 70)
    for nq in QUBIT_RANGE:
        dim = 2 ** nq
        marked_states = np.random.choice(dim, size=min(GROVER_MARKED_STATES, dim), replace=False)
        for ms in marked_states:
            label = f"grover_oracle_q{nq}_m{ms}"
            print(f"\n  {label}")
            U_oracle = generate_grover_oracle(nq, int(ms))
            baselines = compile_baselines_qiskit(U_oracle, nq, label)
            benchmarks[label] = {
                "unitary": U_oracle, "n_qubits": nq,
                "category": "grover_oracle", "marked_state": int(ms),
                "baselines": baselines,
            }

            label = f"grover_iterate_q{nq}_m{ms}"
            print(f"\n  {label}")
            U_iter = generate_grover_iterate(nq, int(ms))
            baselines = compile_baselines_qiskit(U_iter, nq, label)
            benchmarks[label] = {
                "unitary": U_iter, "n_qubits": nq,
                "category": "grover_iterate", "marked_state": int(ms),
                "baselines": baselines,
            }

    # ── 4. Hamiltonian Simulation (Trotter) ──
    print("\n" + "=" * 70)
    print("GENERATING HAMILTONIAN SIMULATION UNITARIES")
    print("=" * 70)
    t_sim = 1.0
    for nq in QUBIT_RANGE:
        for model_name, ham_fn in [("heisenberg", heisenberg_hamiltonian),
                                    ("tfim", tfim_hamiltonian)]:
            H = ham_fn(nq)
            for steps in TROTTER_STEPS:
                label = f"{model_name}_q{nq}_trotter{steps}"
                print(f"\n  {label}")
                U = trotter_unitary(H, t_sim, steps)
                baselines = compile_baselines_qiskit(U, nq, label)
                benchmarks[label] = {
                    "unitary": U, "n_qubits": nq,
                    "category": f"hamsim_{model_name}",
                    "trotter_steps": steps, "sim_time": t_sim,
                    "baselines": baselines,
                }

    # ── 5. KAK Baselines (2-qubit, separate from main split) ──
    print("\n" + "=" * 70)
    print("COMPUTING KAK BASELINES (2-QUBIT SUB-BLOCKS)")
    print("=" * 70)
    kak_results = {}
    two_q_unitaries = generate_haar_unitaries(2, 100)
    for i, U in enumerate(two_q_unitaries):
        label = f"kak_2q_{i:03d}"
        kak_info = kak_baseline_two_qubit(U)
        qiskit_baselines = compile_baselines_qiskit(U, 2, label)
        kak_results[label] = {"unitary": U, "kak": kak_info, "qiskit_baselines": qiskit_baselines}
        if i % 20 == 0:
            print(f"    {label}  KAK 2q_gates={kak_info['two_qubit_gates']}")

    # ── 6. Create FIXED train / val / test split ──
    print("\n" + "=" * 70)
    print("CREATING TRAIN / VAL / TEST SPLIT")
    print("=" * 70)
    train_keys, val_keys, test_keys = create_stratified_split(benchmarks, seed=SEED)
    print(f"  Train: {len(train_keys)}  |  Val: {len(val_keys)}  |  Test: {len(test_keys)}")

    # Show split per stratum
    for split_name, keys in [("train", train_keys), ("val", val_keys), ("test", test_keys)]:
        by_nq = defaultdict(int)
        for k in keys:
            by_nq[benchmarks[k]["n_qubits"]] += 1
        print(f"    {split_name:5s}: " + "  ".join(f"{nq}q={c}" for nq, c in sorted(by_nq.items())))

    # ── 7. Summary Statistics ──
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY (full dataset)")
    print("=" * 70)
    categories = {}
    for label, info in benchmarks.items():
        k = f"{info['category']}_q{info['n_qubits']}"
        if k not in categories:
            categories[k] = {"count": 0, "ibm_opt3_2q": [], "google_opt3_2q": []}
        categories[k]["count"] += 1
        bl = info["baselines"]
        if bl.get("ibm_opt3", {}).get("success"):
            categories[k]["ibm_opt3_2q"].append(bl["ibm_opt3"]["two_qubit_gates"])
        if bl.get("google_opt3", {}).get("success"):
            categories[k]["google_opt3_2q"].append(bl["google_opt3"]["two_qubit_gates"])

    for k, v in sorted(categories.items()):
        ibm_mean = np.mean(v["ibm_opt3_2q"]) if v["ibm_opt3_2q"] else float("nan")
        ggl_mean = np.mean(v["google_opt3_2q"]) if v["google_opt3_2q"] else float("nan")
        print(f"  {k:35s}  n={v['count']:4d}  IBM-opt3 mean 2q={ibm_mean:7.1f}  Google-opt3 mean 2q={ggl_mean:7.1f}")

    # ── 8. Save ──
    payload = {
        "benchmarks": benchmarks,
        "kak_baselines": kak_results,
        # FROZEN split — every training run MUST use these exact keys
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
        "config": {
            "seed": SEED,
            "qubit_range": QUBIT_RANGE,
            "n_haar_samples": N_HAAR_SAMPLES,
            "grover_marked_states": GROVER_MARKED_STATES,
            "trotter_steps": TROTTER_STEPS,
            "qiskit_opt_levels": QISKIT_OPT_LEVELS,
            "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        },
    }

    torch.save(payload, save_path)
    print(f"\n  Saved {len(benchmarks)} benchmark unitaries + {len(kak_results)} KAK baselines")
    print(f"  Split: {len(train_keys)} train / {len(val_keys)} val / {len(test_keys)} test")
    print(f"  → {save_path}")
    print("=" * 70)

    return payload


# ─────────────────────────────────────────────
if __name__ == "__main__":
    build_benchmark_dataset()