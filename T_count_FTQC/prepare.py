#!/usr/bin/env python3
"""
prepare.py — Benchmark Generation and Baseline Computation
===========================================================

Generates benchmark Clifford+T circuits across multiple families,
computes baseline T-counts using all available PyZX optimizers,
and saves everything to benchmarks/ for use by train.py.

Usage:
    python prepare.py [--seed 42] [--output-dir benchmarks]
"""

import argparse
import copy
import json
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyzx as zx
from pyzx.circuit.gates import ZPhase


# ---------------------------------------------------------------------------
# Helper: gate shortcuts (PyZX 0.10 uses ZPhase for T† and S†)
# ---------------------------------------------------------------------------

def add_Tdg(circ, qubit):
    """Add T-dagger gate (phase = -pi/4)."""
    circ.add_gate(ZPhase(qubit, phase=Fraction(-1, 4)))

def add_Sdg(circ, qubit):
    """Add S-dagger gate (phase = -pi/2)."""
    circ.add_gate(ZPhase(qubit, phase=Fraction(-1, 2)))


# ---------------------------------------------------------------------------
# Benchmark generators
# ---------------------------------------------------------------------------

def generate_random_clifford_t(n_qubits: int, depth: int, p_t: float, p_had: float,
                                seed: int) -> zx.Circuit:
    """Generate a random Clifford+T circuit via PyZX."""
    np.random.seed(seed)
    circ = zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=n_qubits, depth=depth, p_had=p_had, p_t=p_t
    )
    return circ


def generate_phase_poly_circuit(n_qubits: int, n_phase_layers: int,
                                 cnots_per_layer: int, seed: int) -> zx.Circuit:
    """Generate a random phase polynomial circuit via PyZX."""
    np.random.seed(seed)
    pp = zx.generate.phase_poly(n_qubits, n_phase_layers, cnots_per_layer)
    # phase_poly returns a CNOT_tracker (subclass of Circuit) — copy to plain Circuit
    circ = zx.Circuit(pp.qubits)
    for g in pp.gates:
        circ.add_gate(g)
    return circ


def make_toffoli(circ, c1, c2, tgt):
    """
    Append a Toffoli gate decomposition (7 T gates) to the circuit.
    Uses the standard Nielsen & Chuang decomposition into Clifford+T.
    """
    circ.add_gate("HAD", tgt)
    circ.add_gate("CNOT", c2, tgt)
    add_Tdg(circ, tgt)
    circ.add_gate("CNOT", c1, tgt)
    circ.add_gate("T", tgt)
    circ.add_gate("CNOT", c2, tgt)
    add_Tdg(circ, tgt)
    circ.add_gate("CNOT", c1, tgt)
    circ.add_gate("T", c2)
    circ.add_gate("T", tgt)
    circ.add_gate("CNOT", c1, c2)
    circ.add_gate("HAD", tgt)
    circ.add_gate("T", c1)
    add_Tdg(circ, c2)
    circ.add_gate("CNOT", c1, c2)


def generate_toffoli_chain(n_toffolis: int) -> zx.Circuit:
    """
    Generate a chain of Toffoli gates on overlapping qubit triples.
    Toffoli(0,1,2), Toffoli(1,2,3), Toffoli(2,3,4), ...
    """
    n_qubits = n_toffolis + 2
    circ = zx.Circuit(n_qubits)
    for i in range(n_toffolis):
        make_toffoli(circ, i, i + 1, i + 2)
    return circ


def generate_toffoli_ladder(n_toffolis: int) -> zx.Circuit:
    """
    Generate Toffoli gates with shared control lines.
    All share qubit 0 as one control.
    """
    n_qubits = n_toffolis + 2
    circ = zx.Circuit(n_qubits)
    for i in range(n_toffolis):
        make_toffoli(circ, 0, i + 1, i + 2)
    return circ


def generate_toffoli_cnot_interleaved(n_blocks: int) -> zx.Circuit:
    """
    Generate interleaved Toffoli + CNOT blocks.
    
    This produces valid Clifford+T circuits with structure similar to
    arithmetic circuits (Toffoli for carry, CNOT for sum). We do NOT
    claim this implements a specific arithmetic function — it is used
    purely as a structured benchmark with known Toffoli T-count.
    
    Qubit layout: 2*n_blocks + 1 qubits.
    """
    n_qubits = 2 * n_blocks + 1
    circ = zx.Circuit(n_qubits)
    carry_q = 2 * n_blocks
    for i in range(n_blocks):
        a_i = i
        b_i = n_blocks + i
        circ.add_gate("CNOT", a_i, b_i)
        make_toffoli(circ, a_i, b_i, carry_q)
        if i < n_blocks - 1:
            circ.add_gate("CNOT", carry_q, n_blocks + i + 1)
    return circ


def generate_all_benchmarks(seed: int = 42) -> list:
    """Generate all benchmark circuits. Returns list of (name, circuit) tuples."""
    benchmarks = []
    rng = np.random.RandomState(seed)

    print("Generating benchmarks...")

    # --- Random Clifford+T circuits ---
    random_configs = [
        (4, 60, 0.35, 0.15, "rand_4q_60d"),
        (4, 100, 0.35, 0.15, "rand_4q_100d"),
        (5, 80, 0.35, 0.15, "rand_5q_80d"),
        (5, 120, 0.35, 0.15, "rand_5q_120d"),
        (6, 80, 0.35, 0.15, "rand_6q_80d"),
        (6, 120, 0.35, 0.15, "rand_6q_120d"),
        (6, 180, 0.35, 0.15, "rand_6q_180d"),
        (8, 100, 0.35, 0.15, "rand_8q_100d"),
        (8, 160, 0.35, 0.15, "rand_8q_160d"),
        (8, 200, 0.35, 0.15, "rand_8q_200d"),
        (10, 120, 0.35, 0.15, "rand_10q_120d"),
        (10, 200, 0.35, 0.15, "rand_10q_200d"),
        (12, 150, 0.35, 0.15, "rand_12q_150d"),
        (12, 250, 0.35, 0.15, "rand_12q_250d"),
        (15, 200, 0.35, 0.15, "rand_15q_200d"),
        (5, 80, 0.50, 0.10, "rand_5q_80d_hiT"),
        (8, 100, 0.50, 0.10, "rand_8q_100d_hiT"),
        (10, 120, 0.50, 0.10, "rand_10q_120d_hiT"),
        (6, 100, 0.20, 0.25, "rand_6q_100d_loT"),
        (8, 120, 0.20, 0.25, "rand_8q_120d_loT"),
    ]

    for n_q, depth, p_t, p_had, label in random_configs:
        s = rng.randint(0, 100000)
        try:
            circ = generate_random_clifford_t(n_q, depth, p_t, p_had, seed=s)
            benchmarks.append((f"random/{label}", circ))
        except Exception as e:
            print(f"  WARNING: Failed {label}: {e}")

    # --- Phase polynomial circuits ---
    phase_poly_configs = [
        (4, 10, 2, "pp_4q_10l"),
        (4, 20, 2, "pp_4q_20l"),
        (5, 15, 2, "pp_5q_15l"),
        (5, 25, 3, "pp_5q_25l"),
        (6, 15, 3, "pp_6q_15l"),
        (6, 30, 3, "pp_6q_30l"),
        (8, 20, 3, "pp_8q_20l"),
        (8, 35, 4, "pp_8q_35l"),
        (10, 25, 4, "pp_10q_25l"),
        (10, 40, 4, "pp_10q_40l"),
        (12, 30, 4, "pp_12q_30l"),
    ]

    for n_q, n_layers, cnots, label in phase_poly_configs:
        s = rng.randint(0, 100000)
        try:
            circ = generate_phase_poly_circuit(n_q, n_layers, cnots, seed=s)
            benchmarks.append((f"phase_poly/{label}", circ))
        except Exception as e:
            print(f"  WARNING: Failed {label}: {e}")

    # --- Toffoli chain circuits ---
    for n_tof in [1, 2, 3, 4, 5, 6, 8, 10]:
        try:
            circ = generate_toffoli_chain(n_tof)
            benchmarks.append((f"toffoli_chain/tof_chain_{n_tof}", circ))
        except Exception as e:
            print(f"  WARNING: Failed tof_chain_{n_tof}: {e}")

    # --- Toffoli ladder circuits ---
    for n_tof in [2, 3, 4, 5, 8]:
        try:
            circ = generate_toffoli_ladder(n_tof)
            benchmarks.append((f"toffoli_ladder/tof_ladder_{n_tof}", circ))
        except Exception as e:
            print(f"  WARNING: Failed tof_ladder_{n_tof}: {e}")

    # --- Toffoli-CNOT interleaved (structured, arithmetic-like) ---
    for n_blocks in [2, 3, 4, 5, 6, 8]:
        try:
            circ = generate_toffoli_cnot_interleaved(n_blocks)
            benchmarks.append((f"toffoli_interleaved/tof_inter_{n_blocks}", circ))
        except Exception as e:
            print(f"  WARNING: Failed tof_inter_{n_blocks}: {e}")

    return benchmarks


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def count_gates(circ: zx.Circuit) -> dict:
    """Count gate types. Uses zx.tcount() for authoritative T-count."""
    stats = {"t_count": zx.tcount(circ), "cnot_count": 0, "had_count": 0,
             "total_gates": len(circ.gates), "s_count": 0}
    for gate in circ.gates:
        name = gate.name
        if name in ("CNOT", "CX", "CZ"):
            stats["cnot_count"] += 1
        elif name in ("HAD", "H"):
            stats["had_count"] += 1
        elif name in ("S",):
            stats["s_count"] += 1
    return stats


def compute_baseline_full_reduce(circ: zx.Circuit) -> dict:
    """Baseline: PyZX full_reduce -> extract -> basic_optimization."""
    t0 = time.time()
    try:
        g = circ.to_graph()
        zx.simplify.full_reduce(g)
        c_opt = zx.extract_circuit(g.copy())
        c_opt = c_opt.to_basic_gates()
        try:
            c_opt = zx.optimize.basic_optimization(c_opt)
        except Exception:
            pass
        elapsed = time.time() - t0

        verified = None
        if circ.qubits <= 6:
            try:
                verified = circ.verify_equality(c_opt)
            except Exception:
                verified = "error"

        stats = count_gates(c_opt)
        stats["time_s"] = round(elapsed, 4)
        stats["verified"] = verified
        stats["success"] = True
        return stats
    except Exception as e:
        return {"success": False, "error": str(e), "time_s": round(time.time() - t0, 4)}


def compute_baseline_full_optimize(circ: zx.Circuit) -> dict:
    """Baseline: PyZX full_optimize."""
    t0 = time.time()
    try:
        c_opt = zx.optimize.full_optimize(circ.to_basic_gates().copy())
        elapsed = time.time() - t0

        verified = None
        if circ.qubits <= 6:
            try:
                verified = circ.verify_equality(c_opt)
            except Exception:
                verified = "error"

        stats = count_gates(c_opt)
        stats["time_s"] = round(elapsed, 4)
        stats["verified"] = verified
        stats["success"] = True
        return stats
    except Exception as e:
        return {"success": False, "error": str(e), "time_s": round(time.time() - t0, 4)}


def compute_baseline_phase_block(circ: zx.Circuit) -> dict:
    """Baseline: PyZX phase_block_optimize only."""
    t0 = time.time()
    try:
        c_opt = zx.optimize.phase_block_optimize(circ.to_basic_gates().copy())
        elapsed = time.time() - t0

        verified = None
        if circ.qubits <= 6:
            try:
                verified = circ.verify_equality(c_opt)
            except Exception:
                verified = "error"

        stats = count_gates(c_opt)
        stats["time_s"] = round(elapsed, 4)
        stats["verified"] = verified
        stats["success"] = True
        return stats
    except Exception as e:
        return {"success": False, "error": str(e), "time_s": round(time.time() - t0, 4)}


def compute_all_baselines(circ: zx.Circuit) -> dict:
    """Compute all baselines for a single circuit."""
    original_stats = count_gates(circ.to_basic_gates())

    return {
        "original": {**original_stats, "success": True, "time_s": 0.0, "verified": True},
        "full_reduce": compute_baseline_full_reduce(circ),
        "full_optimize": compute_baseline_full_optimize(circ),
        "phase_block": compute_baseline_phase_block(circ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate benchmarks and compute baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "circuits").mkdir(parents=True, exist_ok=True)

    print(f"PyZX version: {zx.__version__}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}\n")

    # Generate benchmarks
    print("=" * 60)
    print("GENERATING BENCHMARKS")
    print("=" * 60)
    benchmarks = generate_all_benchmarks(seed=args.seed)
    print(f"\nGenerated {len(benchmarks)} benchmark circuits.\n")

    # Compute baselines
    print("=" * 60)
    print("COMPUTING BASELINES")
    print("=" * 60)

    all_results = {}
    summary = {"total": 0, "families": {}}

    for i, (name, circ) in enumerate(benchmarks):
        family = name.split("/")[0]
        label = name.split("/")[1] if "/" in name else name
        orig_t = zx.tcount(circ)

        print(f"\n[{i+1}/{len(benchmarks)}] {name}")
        print(f"  Qubits: {circ.qubits}, Original T-count: {orig_t}")

        # Save QASM
        circuit_path = output_dir / "circuits" / f"{label}.qasm"
        try:
            qasm = circ.to_basic_gates().to_qasm()
            circuit_path.write_text(qasm)
        except Exception as e:
            print(f"  WARNING: Could not save QASM: {e}")

        # Compute baselines
        baselines = compute_all_baselines(circ)

        for bname in ["full_reduce", "full_optimize", "phase_block"]:
            bdata = baselines[bname]
            if bdata["success"]:
                t = bdata["t_count"]
                delta = orig_t - t
                pct = (delta / orig_t * 100) if orig_t > 0 else 0.0
                v_str = ""
                if bdata.get("verified") is not None:
                    v_str = f" [{'OK' if bdata['verified'] == True else 'FAIL'}]"
                print(f"  {bname:20s}: T={t:4d}  (Δ={delta:+4d}, {pct:+5.1f}%)  "
                      f"t={bdata['time_s']:.3f}s{v_str}")
            else:
                print(f"  {bname:20s}: FAILED — {bdata.get('error', 'unknown')}")

        all_results[name] = {
            "name": name, "family": family, "qubits": circ.qubits,
            "baselines": baselines,
        }
        summary["total"] += 1
        summary["families"].setdefault(family, 0)
        summary["families"][family] += 1

    # Save
    results_path = output_dir / "baselines.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total benchmarks: {summary['total']}")
    for fam, count in sorted(summary["families"].items()):
        print(f"  {fam}: {count}")

    n_valid = 0
    wins = {"full_reduce": 0, "full_optimize": 0, "phase_block": 0}
    for entry in all_results.values():
        orig_t = entry["baselines"]["original"]["t_count"]
        if orig_t == 0:
            continue
        n_valid += 1
        for bname in wins:
            bdata = entry["baselines"][bname]
            if bdata.get("success") and bdata["t_count"] < orig_t:
                wins[bname] += 1

    print(f"\nBaseline improvement rates (out of {n_valid} circuits with T>0):")
    for bname, w in wins.items():
        pct = (w / n_valid * 100) if n_valid > 0 else 0
        print(f"  {bname}: {w}/{n_valid} ({pct:.1f}%)")

    n_same = sum(1 for e in all_results.values()
                 if e["baselines"]["full_reduce"].get("success")
                 and e["baselines"]["full_optimize"].get("success")
                 and e["baselines"]["full_reduce"]["t_count"] == e["baselines"]["full_optimize"]["t_count"])
    print(f"\nfull_reduce == full_optimize on {n_same}/{summary['total']} benchmarks")

    config = {"seed": args.seed, "pyzx_version": zx.__version__,
              "n_benchmarks": len(benchmarks), "families": summary["families"]}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nBaselines saved to {results_path}")
    print(f"Config saved to {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()