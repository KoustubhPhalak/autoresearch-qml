#!/usr/bin/env python3
"""
train.py — Three-Stage T-Count Optimization Pipeline
=====================================================

Stage 1: ZX-calculus preconditioning (beam search over rewrite sequences)
Stage 2: Hadamard-segmented phase-polynomial resynthesis (core contribution)
Stage 3: Local cleanup (commutation, cancellation, peephole)

Reads benchmarks from benchmarks/ (produced by prepare.py).
Saves results to results/.

Usage:
    python train.py [--benchmarks-dir benchmarks] [--results-dir results]
                    [--beam-width 5] [--max-beam-depth 20]
"""

import argparse
import copy
import json
import os
import random
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import pyzx as zx
import pyzx.simplify as simp

# -----------------------------------------------------------------------
# Fix random seeds for reproducibility.
# NOTE: PYTHONHASHSEED must be set at process launch; set it to 0 by running:
#   PYTHONHASHSEED=0 python train.py
# We also seed numpy and Python's random module here.
# PYTHONHASHSEED=0 is required for deterministic dict/set iteration order in
# PyZX graph operations (ZX rewrite rules iterate over vertices in hash order).
# -----------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ===========================================================================
# GRPO Policy for ZX Rewrite Rule Optimization (iter13)
# ===========================================================================

class GRPOZXPolicy:
    """
    GRPO (Group Relative Policy Optimization) policy for ZX rewrite rule selection.

    Learns which sequences of ZX rewrite rules tend to produce circuits that
    TODD can optimize more aggressively. Updated online as benchmarks are processed:
    after each benchmark, the K rollouts' T-count rewards (group-relative) update
    rule probabilities — rules used in high-reward rollouts are preferred more.

    This is a policy-gradient method where:
    - Policy: softmax distribution over the 8 ZX rewrite rules
    - Prompt: the current circuit's ZX graph
    - Response: a sequence of rule applications (rollout)
    - Reward: T-count reduction achieved (original_T - TODD_output_T)
    - Advantage: (reward - mean(rewards)) / std(rewards)  [group-relative]
    - Update: logit[r] += lr * advantage * freq(r in rollout)
    """

    def __init__(self, n_rules: int, lr: float = 0.05, temperature: float = 1.0):
        self.n_rules = n_rules
        self.lr = lr
        self.temperature = temperature
        self.logits = np.zeros(n_rules)   # All rules start equally likely

    def get_probs(self) -> np.ndarray:
        scaled = self.logits / max(self.temperature, 0.1)
        probs = np.exp(scaled - np.max(scaled))
        return probs / (probs.sum() + 1e-12)

    def sample_action(self) -> int:
        """Sample a rule index from the current policy distribution."""
        probs = self.get_probs()
        return int(np.random.choice(self.n_rules, p=probs))

    def freeze(self):
        """Freeze the policy: disable further updates (for clean evaluation)."""
        self._frozen = True

    def is_frozen(self) -> bool:
        return getattr(self, "_frozen", False)

    def update(self, rollout_histories: list, rewards: list):
        """
        GRPO update: group-relative policy gradient.

        For each rollout k: advantage_k = (reward_k - mean) / std
        Update logit[r] += lr * advantage_k * freq(r in rollout k).

        Does nothing if the policy is frozen (evaluation mode).
        """
        if self.is_frozen():
            return  # Policy frozen: evaluation mode, no updates

        if len(rewards) < 2:
            return
        rewards_arr = np.array(rewards, dtype=float)
        mean_r = float(np.mean(rewards_arr))
        std_r = float(np.std(rewards_arr)) + 1e-8

        for history, reward in zip(rollout_histories, rewards):
            if not history:
                continue
            advantage = (reward - mean_r) / std_r
            rule_counts = np.zeros(self.n_rules)
            for rule_idx in history:
                rule_counts[rule_idx] += 1
            # Weight by frequency (normalize by rollout length)
            freq = rule_counts / (len(history) + 1e-8)
            self.logits += self.lr * advantage * freq

        # Stabilize: center and clip to prevent extreme concentrations
        self.logits -= np.mean(self.logits)
        self.logits = np.clip(self.logits, -3.0, 3.0)

    def top_rules(self, rules: list) -> str:
        """Return top-3 rule names by current probability (for logging)."""
        probs = self.get_probs()
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return ", ".join(f"{rules[i][0]}:{p:.2f}" for i, p in ranked)


def grpo_zx_search(circ: zx.Circuit, policy: GRPOZXPolicy,
                   K: int = 4, max_depth: int = 15,
                   verbose: bool = False) -> tuple:
    """
    GRPO-guided stochastic search over ZX rewrite sequences.

    Generates K rollouts where at each step a rewrite rule is sampled from
    the current policy. After all K rollouts, updates the policy using
    group-relative advantages. Returns the best circuit found.

    This complements the deterministic beam search (Stage 1) with stochastic
    exploration: beam search is greedy and exploitation-focused; GRPO rollouts
    can discover rule sequences that the proxy score would not prioritize but
    that lead to better TODD reductions.
    """
    g_init = circ.to_graph()
    simp.to_graph_like(g_init)

    try:
        baseline_t = zx.tcount(circ.to_basic_gates())
    except Exception:
        baseline_t = 0

    rollout_histories = []
    rollout_rewards = []
    best_t = float("inf")
    best_circ = None
    # Also track the best pre-TODD circuit (basic_opt-only, no TODD).
    # pbo applied to this circuit can find lower T-counts on some circuit families
    # where TODD's rearrangement destroys the structure pbo needs.
    best_pretod_t = float("inf")
    best_pretod_circ = None

    for k in range(K):
        g = copy.deepcopy(g_init)
        history = []

        # Stochastic rollout: sample rules from policy distribution
        for step in range(max_depth):
            rule_idx = policy.sample_action()
            _, rule_fn = ZX_REWRITE_RULES[rule_idx]
            try:
                n = rule_fn(g)
                if n > 0:
                    history.append(rule_idx)
            except Exception:
                pass

        # Extract circuit and apply basic_opt → TODD → basic_opt
        try:
            g_copy = copy.deepcopy(g)
            simp.full_reduce(g_copy)
            c_extracted = zx.extract_circuit(g_copy)
            c_extracted = c_extracted.to_basic_gates()
            c_opt = zx.optimize.basic_optimization(c_extracted)

            # Track best pre-TODD circuit for Path I2
            t_pretod = zx.tcount(c_opt)
            if t_pretod < best_pretod_t:
                best_pretod_t = t_pretod
                best_pretod_circ = c_opt

            c_s2, _, _ = stage2_phase_poly_resynthesis(c_opt, verbose=False)
            c_final = zx.optimize.basic_optimization(c_s2.to_basic_gates())
            t = zx.tcount(c_final)

            rollout_histories.append(history)
            rollout_rewards.append(float(baseline_t - t))

            if t < best_t:
                best_t = t
                best_circ = c_final
        except Exception:
            rollout_histories.append(history)
            rollout_rewards.append(0.0)

    # GRPO policy update: rules used in high-reward rollouts get higher probability
    policy.update(rollout_histories, rollout_rewards)

    if verbose and rollout_rewards:
        print(f"    GRPO rollouts: best_T={best_t}, rewards={[int(r) for r in rollout_rewards]}, "
              f"top rules: [{policy.top_rules(ZX_REWRITE_RULES)}]")

    return best_circ, best_t, best_pretod_circ


# ===========================================================================
# Stage 1: ZX-Calculus Preconditioning with Beam Search
# ===========================================================================

# Available ZX rewrite rules for beam search
ZX_REWRITE_RULES = [
    ("spider_simp", simp.spider_simp),
    ("id_simp", simp.id_simp),
    ("pivot_simp", simp.pivot_simp),
    ("lcomp_simp", simp.lcomp_simp),
    ("bialg_simp", simp.bialg_simp),
    ("gadget_simp", simp.gadget_simp),
    ("pivot_gadget_simp", simp.pivot_gadget_simp),
    ("pivot_boundary_simp", simp.pivot_boundary_simp),
]

# Shared GRPO policy — persists and learns across all benchmarks in a run.
# Initialized after ZX_REWRITE_RULES so n_rules is correct.
GRPO_POLICY = GRPOZXPolicy(n_rules=len(ZX_REWRITE_RULES), lr=0.05, temperature=1.0)


def proxy_t_score(g) -> int:
    """
    Cheap proxy for T-count in a ZX graph.

    Uses a combined score: (non-Clifford phase count) + 0.01 * (edge count).
    The non-Clifford phase count is the primary driver (correlates with T-count);
    the edge count breaks ties, preferring sparser graphs that tend to yield
    circuits with better phase-polynomial structure for TODD.
    Both terms are O(|V|+|E|) and are fast to compute.

    Non-Clifford phases are those not in {0, 1/2, 1, 3/2} (i.e., not multiples of π/2).
    """
    clifford_phases = {Fraction(0), Fraction(1, 2), Fraction(1, 1), Fraction(3, 2)}
    nc_count = 0
    for v in g.vertices():
        p = g.phase(v)
        if isinstance(p, Fraction) and p not in clifford_phases:
            nc_count += 1
        elif not isinstance(p, Fraction):
            nc_count += 1
    edge_count = g.num_edges()
    # Combined score: primary = non-Clifford phase count; secondary = edge count.
    # The 100× weight ensures nc_count dominates while edge count breaks ties.
    return int(nc_count * 100 + edge_count)


def zx_beam_search(circ: zx.Circuit, beam_width: int = 5, max_depth: int = 20,
                   verbose: bool = False) -> list:
    """
    Beam search over ZX rewrite rule orderings.
    
    Returns a list of candidate ZX graphs, sorted by proxy T-score (ascending).
    The caller should extract circuits from the best candidates.
    
    Each beam state is (graph, score, rule_history).
    At each step, we try all rewrite rules on all beam states and keep the top-k.
    """
    g_init = circ.to_graph()
    simp.to_graph_like(g_init)

    init_score = proxy_t_score(g_init)
    beam = [(copy.deepcopy(g_init), init_score, [])]
    best_score = init_score
    best_graphs = [copy.deepcopy(g_init)]
    stale_steps = 0

    for step in range(max_depth):
        candidates = []

        for g, score, history in beam:
            for rule_name, rule_fn in ZX_REWRITE_RULES:
                g_copy = copy.deepcopy(g)
                try:
                    n_rewrites = rule_fn(g_copy)
                except Exception:
                    continue

                if n_rewrites > 0:
                    new_score = proxy_t_score(g_copy)
                    candidates.append((g_copy, new_score, history + [rule_name]))

            # Also keep current state as a candidate (allow "no-op" to preserve good states)
            candidates.append((copy.deepcopy(g), score, history + ["hold"]))

        if not candidates:
            break

        # Sort by proxy score, keep top beam_width
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]

        # Track best
        if beam[0][1] < best_score:
            best_score = beam[0][1]
            best_graphs = [copy.deepcopy(beam[0][0])]
            stale_steps = 0
        else:
            stale_steps += 1

        if verbose:
            scores = [s for _, s, _ in beam]
            print(f"    Step {step+1}: best_proxy={best_score}, "
                  f"beam_scores={scores[:3]}, rules={beam[0][2][-3:]}")

        # Early stopping if no improvement for a while
        if stale_steps >= 5:
            break

    # Also run full_reduce as a candidate
    g_fr = circ.to_graph()
    simp.full_reduce(g_fr)
    fr_score = proxy_t_score(g_fr)
    best_graphs.append(g_fr)

    if verbose:
        print(f"    Beam search best proxy: {best_score}, full_reduce proxy: {fr_score}")

    return best_graphs, beam


def stage1_zx_precondition(circ: zx.Circuit, beam_width: int = 5,
                            max_depth: int = 20, verbose: bool = False) -> list:
    """
    Stage 1: ZX preconditioning.
    
    Extracts circuits from ALL beam top-k candidates plus full_reduce,
    compares actual T-counts, and returns sorted candidates.
    """
    t0 = time.time()
    best_graphs, beam = zx_beam_search(circ, beam_width, max_depth, verbose=verbose)

    # Extract circuits from ALL beam candidates (top-k by proxy score)
    candidates = []
    seen_t_counts = set()

    # First: extract from beam top-k (these were selected by proxy score)
    for g, proxy_score, history in beam:
        try:
            g_copy = copy.deepcopy(g)
            simp.full_reduce(g_copy)
            c_extracted = zx.extract_circuit(g_copy)
            c_extracted = c_extracted.to_basic_gates()
            t_count = zx.tcount(c_extracted)
            if t_count not in seen_t_counts:
                candidates.append((c_extracted, t_count))
                seen_t_counts.add(t_count)
        except Exception:
            continue

    # Also extract from best_graphs (which includes the single-best from search)
    for g in best_graphs:
        try:
            g_copy = copy.deepcopy(g)
            simp.full_reduce(g_copy)
            c_extracted = zx.extract_circuit(g_copy)
            c_extracted = c_extracted.to_basic_gates()
            t_count = zx.tcount(c_extracted)
            if t_count not in seen_t_counts:
                candidates.append((c_extracted, t_count))
                seen_t_counts.add(t_count)
        except Exception:
            continue

    # Also include direct full_reduce as fallback
    try:
        g_fr = circ.to_graph()
        simp.full_reduce(g_fr)
        c_fr = zx.extract_circuit(g_fr.copy())
        c_fr = c_fr.to_basic_gates()
        t_fr = zx.tcount(c_fr)
        if t_fr not in seen_t_counts:
            candidates.append((c_fr, t_fr))
    except Exception:
        pass

    # Sort by actual T-count (not proxy), keep best
    candidates.sort(key=lambda x: x[1])

    elapsed = time.time() - t0

    if verbose:
        if candidates:
            print(f"    Stage 1: {len(candidates)} unique candidates evaluated, "
                  f"best T={candidates[0][1]}, time={elapsed:.2f}s")
        else:
            print(f"    Stage 1: no valid candidates, time={elapsed:.2f}s")

    return candidates, elapsed


# ===========================================================================
# Stage 2: Phase-Polynomial Resynthesis
# ===========================================================================

def segment_by_hadamards(circ: zx.Circuit) -> list:
    """
    Partition circuit gates into Hadamard-free segments separated by Hadamard layers.
    
    Returns list of segments, where each segment is a dict:
    {
        "type": "phase_poly" | "hadamard_layer",
        "gates": [list of gates],
        "start_idx": int,
        "end_idx": int,
    }
    """
    segments = []
    current_gates = []
    current_start = 0

    for i, gate in enumerate(circ.gates):
        if gate.name in ("HAD", "H"):
            if current_gates:
                segments.append({
                    "type": "phase_poly",
                    "gates": current_gates,
                    "start_idx": current_start,
                    "end_idx": i - 1,
                })
                current_gates = []
            # Add Hadamard as its own layer
            segments.append({
                "type": "hadamard_layer",
                "gates": [gate],
                "start_idx": i,
                "end_idx": i,
            })
            current_start = i + 1
        else:
            current_gates.append(gate)

    if current_gates:
        segments.append({
            "type": "phase_poly",
            "gates": current_gates,
            "start_idx": current_start,
            "end_idx": len(circ.gates) - 1,
        })

    return segments


def count_t_in_gates(gates: list) -> int:
    """Count T/T† gates in a gate list."""
    count = 0
    for g in gates:
        if g.name == "T":
            count += 1
        elif g.name == "ZPhase":
            # Check if phase is an odd multiple of pi/4
            p = g.phase
            if isinstance(p, Fraction):
                # T-like if 4*p is odd
                val = (p * 4) % 2
                if val != 0:
                    count += 1
    return count


def permutation_to_swaps(perm: dict) -> list:
    """
    Decompose a permutation (dict mapping qubit -> qubit) into a sequence of
    transpositions (2-element swaps). Each transposition (a, b) means SWAP a and b.
    
    Algorithm: decompose into disjoint cycles, then each cycle of length k
    becomes k-1 transpositions. E.g. cycle (a,b,c) = SWAP(a,b) then SWAP(a,c).
    """
    visited = set()
    swaps = []
    
    for start in sorted(perm.keys()):
        if start in visited or perm.get(start, start) == start:
            continue
        
        # Trace the cycle starting from 'start'
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = perm[current]
        
        # Cycle of length k needs k-1 transpositions
        # (a, b, c, d) -> SWAP(a,d), SWAP(a,c), SWAP(a,b)
        # This maps: a->b, b->c, c->d, d->a correctly
        for i in range(len(cycle) - 1, 0, -1):
            swaps.append((cycle[0], cycle[i]))
    
    return swaps


def optimize_phase_poly_segment(gates: list, n_qubits: int) -> list:
    """
    Optimize a Hadamard-free (phase-polynomial) segment using PyZX's todd_simp.

    IMPORTANT: Gate ordering must NOT be shuffled — CNOT and phase gates do not
    commute freely. The input gate list represents a specific phase polynomial
    and must be presented to todd_simp in its correct order.

    Future improvement: implement proper ILP/SAT-based exact optimization that
    works on the algebraic phase polynomial representation, independent of gate order.
    """
    if not gates:
        return gates

    original_t = count_t_in_gates(gates)
    if original_t == 0:
        return gates

    # Filter: todd_simp only handles CNOT, T, S, Z, ZPhase (no HAD)
    has_had = any(g.name in ("HAD", "H") for g in gates)
    if has_had:
        return gates  # Cannot apply todd_simp to segments with Hadamards

    try:
        optimized_gates, perm = zx.optimize.todd_simp(list(gates), n_qubits)
        new_t = count_t_in_gates(optimized_gates)

        if new_t <= original_t:
            final_gates = list(optimized_gates)
            swaps = permutation_to_swaps(perm)
            for q_a, q_b in swaps:
                # SWAP(a,b) = CNOT(a,b) CNOT(b,a) CNOT(a,b)
                final_gates.append(zx.circuit.gates.CNOT(q_a, q_b))
                final_gates.append(zx.circuit.gates.CNOT(q_b, q_a))
                final_gates.append(zx.circuit.gates.CNOT(q_a, q_b))
            return final_gates
        else:
            return gates
    except Exception:
        return gates


def stage2_phase_poly_resynthesis(circ: zx.Circuit, verbose: bool = False) -> tuple:
    """
    Stage 2: Single-pass Hadamard-segmented phase-polynomial resynthesis.

    Single-pass TODD is empirically better than iterative passes because
    the output SWAP decomposition from todd_simp adds extra CNOT gates
    that pollute subsequent passes. The input circuit should already have
    basic_optimization applied (done in run_pipeline before calling Stage 2).
    """
    t0 = time.time()
    c_basic = circ.to_basic_gates()
    segments = segment_by_hadamards(c_basic)

    n_phase_segments = sum(1 for s in segments if s["type"] == "phase_poly")
    n_t_segments = sum(1 for s in segments if s["type"] == "phase_poly"
                       and any(g.name in ("T", "T*") for g in s["gates"]))

    if verbose:
        print(f"    Stage 2: {len(segments)} segments ({n_phase_segments} phase-poly, "
              f"{n_t_segments} with T gates)")

    optimized_segments = []
    segment_improvements = []

    for seg in segments:
        if seg["type"] == "hadamard_layer":
            optimized_segments.append(seg["gates"])
        elif seg["type"] == "phase_poly":
            original_gates = seg["gates"]
            orig_t = count_t_in_gates(original_gates)
            optimized_gates = optimize_phase_poly_segment(original_gates, circ.qubits)
            new_t = count_t_in_gates(optimized_gates)
            optimized_segments.append(optimized_gates)
            if orig_t > 0:
                segment_improvements.append({
                    "original_t": orig_t,
                    "optimized_t": new_t,
                    "delta": orig_t - new_t,
                })

    new_circ = zx.Circuit(circ.qubits)
    for gate_list in optimized_segments:
        for gate in gate_list:
            new_circ.add_gate(gate)

    elapsed = time.time() - t0
    total_delta = sum(s["delta"] for s in segment_improvements)

    if verbose:
        print(f"    Stage 2: Δ_T={total_delta} across {len(segment_improvements)} segments, "
              f"time={elapsed:.2f}s")

    return new_circ, elapsed, segment_improvements


# ===========================================================================
# Stage 3: Local Cleanup
# ===========================================================================

def stage3_local_cleanup(circ: zx.Circuit, verbose: bool = False) -> tuple:
    """
    Stage 3: Local cleanup.
    
    - basic_optimization (commutation, cancellation)
    - Additional peephole optimization could be added here
    """
    t0 = time.time()

    try:
        c_opt = zx.optimize.basic_optimization(circ.to_basic_gates())
    except Exception:
        c_opt = circ.to_basic_gates()

    elapsed = time.time() - t0

    if verbose:
        print(f"    Stage 3: cleanup T={zx.tcount(c_opt)}, time={elapsed:.2f}s")

    return c_opt, elapsed


# ===========================================================================
# Full Pipeline
# ===========================================================================

def compute_t_depth(circ: zx.Circuit) -> int:
    """
    Compute T-depth: the minimum number of T-layers when T gates are parallelized.
    
    Simulates a simple ASAP (as-soon-as-possible) schedule: each qubit has a "time"
    counter. Two-qubit gates synchronize their qubits. T gates advance time by 1.
    T-depth = max time across all qubits.
    """
    qubit_time = [0] * circ.qubits
    
    for gate in circ.gates:
        name = gate.name
        is_t = False
        
        if name == "T":
            is_t = True
        elif name == "ZPhase" and hasattr(gate, 'phase'):
            p = gate.phase
            if isinstance(p, Fraction):
                if (p * 4) % 2 != 0:
                    is_t = True
        
        if hasattr(gate, 'control') and gate.control is not None:
            # Two-qubit gate: synchronize
            sync_time = max(qubit_time[gate.control], qubit_time[gate.target])
            qubit_time[gate.control] = sync_time
            qubit_time[gate.target] = sync_time
            if is_t:
                qubit_time[gate.target] += 1
        elif hasattr(gate, 'target'):
            if is_t:
                qubit_time[gate.target] += 1
    
    return max(qubit_time) if qubit_time else 0


def count_gates(circ: zx.Circuit) -> dict:
    """Count various gate types in a circuit. Uses zx.tcount for authoritative T-count."""
    stats = {
        "t_count": zx.tcount(circ),
        "t_depth": compute_t_depth(circ),
        "cnot_count": 0,
        "had_count": 0,
        "total_gates": len(circ.gates),
    }
    for gate in circ.gates:
        name = gate.name
        if name in ("CNOT", "CX", "CZ"):
            stats["cnot_count"] += 1
        elif name in ("HAD", "H"):
            stats["had_count"] += 1
    return stats


def _apply_circuit_statevec(circ: zx.Circuit, psi: np.ndarray) -> np.ndarray:
    """
    Apply a Clifford+T circuit (in basic_gates form: CNOT, HAD, T only) to a
    state vector psi of size 2^n. Fully vectorized, no Python loops over states.
    O(gates × 2^n) instead of O(gates × 4^n) needed for full unitary computation.

    PyZX uses big-endian qubit ordering: qubit q corresponds to bit position (n-1-q)
    in the state index. E.g., for n=2: index 2 = |10⟩ means qubit 0 = 1, qubit 1 = 0.
    """
    n = circ.qubits
    idx = np.arange(len(psi))
    for gate in circ.gates:
        name = type(gate).__name__
        if name == "T":
            q = gate.target
            q_bit = n - 1 - q  # big-endian: qubit q → bit position (n-1-q)
            is_adj = getattr(gate, "adjoint", False)
            phase = np.exp(-1j * np.pi / 4) if is_adj else np.exp(1j * np.pi / 4)
            mask = ((idx >> q_bit) & 1).astype(bool)
            psi = psi.copy()
            psi[mask] *= phase
        elif name == "ZPhase":
            # ZPhase(phase=p) applies e^{i*p*π} to |1⟩ component (Z-rotation by p*π)
            q = gate.target
            q_bit = n - 1 - q
            p = float(gate.phase)  # phase in units of π
            phase = np.exp(1j * np.pi * p)
            mask = ((idx >> q_bit) & 1).astype(bool)
            psi = psi.copy()
            psi[mask] *= phase
        elif name == "HAD":
            q = gate.target
            q_bit = n - 1 - q
            lower = idx[~((idx >> q_bit) & 1).astype(bool)]
            upper = lower | (1 << q_bit)
            psi = psi.copy()
            s2 = 1.0 / np.sqrt(2)
            a, b = psi[lower], psi[upper]
            psi[lower] = s2 * (a + b)
            psi[upper] = s2 * (a - b)
        elif name == "CNOT":
            ctrl, tgt = gate.control, gate.target
            ctrl_bit = n - 1 - ctrl
            tgt_bit = n - 1 - tgt
            ctrl_mask = ((idx >> ctrl_bit) & 1).astype(bool)
            perm = idx.copy()
            perm[ctrl_mask] ^= (1 << tgt_bit)
            psi = psi[perm]
        elif name == "CZ":
            # CZ: apply -1 phase when both control and target are |1⟩
            ctrl, tgt = gate.control, gate.target
            ctrl_bit = n - 1 - ctrl
            tgt_bit = n - 1 - tgt
            both_one = (((idx >> ctrl_bit) & 1) & ((idx >> tgt_bit) & 1)).astype(bool)
            psi = psi.copy()
            psi[both_one] *= -1.0
        elif name in ("Z", "S", "NOT", "X"):
            # Single-qubit Pauli/phase gates expressible as ZPhase:
            # Z: phase π → e^{iπ}=-1 on |1⟩; S: phase π/2; X/NOT: bit-flip
            q = gate.target
            q_bit = n - 1 - q
            if name == "Z":
                phase_factor = -1.0 + 0j
                mask = ((idx >> q_bit) & 1).astype(bool)
                psi = psi.copy()
                psi[mask] *= phase_factor
            elif name == "S":
                is_adj = getattr(gate, "adjoint", False)
                phase_factor = np.exp(-1j * np.pi / 2) if is_adj else np.exp(1j * np.pi / 2)
                mask = ((idx >> q_bit) & 1).astype(bool)
                psi = psi.copy()
                psi[mask] *= phase_factor
            elif name in ("NOT", "X"):
                # Bit-flip: swap |0⟩ and |1⟩ components for qubit q
                perm = idx.copy()
                perm ^= (1 << q_bit)
                psi = psi[perm]
    return psi


def _statevec_equiv(c1: zx.Circuit, c2: zx.Circuit, k: int = 5,
                    atol: float = 1e-6) -> str:
    """
    Randomized state-vector equivalence check for 11-15 qubit circuits.
    Applies both circuits to k random unit vectors; verifies agreement up to
    global phase. False-positive probability ≈ (1 / 2^(n*k)) — negligible.
    Returns "verified", "failed", or "unverified".
    """
    n = c1.qubits
    dim = 2 ** n
    try:
        c1_basic = c1.to_basic_gates()
        c2_basic = c2.to_basic_gates()
        for _ in range(k):
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi /= np.linalg.norm(psi)
            out1 = _apply_circuit_statevec(c1_basic, psi.copy())
            out2 = _apply_circuit_statevec(c2_basic, psi.copy())
            # Find global phase from largest component
            idx_max = np.argmax(np.abs(out1))
            if np.abs(out1[idx_max]) < 1e-12:
                return "unverified"
            phase = out2[idx_max] / out1[idx_max]
            if abs(abs(phase) - 1.0) > 0.01:
                return "failed"
            diff = np.linalg.norm(out2 - phase * out1)
            if diff > atol:
                return "failed"
        return "verified"
    except Exception:
        return "unverified"


def verify_equivalence(original: zx.Circuit, optimized: zx.Circuit) -> str:
    """
    Verify circuit equivalence.

    Returns:
        "verified"    — circuits are provably equivalent
        "failed"      — circuits are provably NOT equivalent (result must be discarded)
        "unverified"  — verification could not be completed (too large or error)

    For ≤10 qubits: exact unitary comparison, including equality up to global phase.
    For 11-15 qubits: randomized state-vector sampling (k=5 random states; false-
      positive probability negligible: ≈ (1/2^(n*k))).
    For >15 qubits: ZX-calculus identity test (incomplete — may return "unverified").

    phase_block_optimize (TODD) can produce circuits equivalent up to global phase,
    which has no physical effect. We check |tr(U†V)| = dim for ≤10q, and phase-
    invariant overlap for 11-15q.

    NOTE: full_reduce is INCOMPLETE for Clifford+T circuits. A composition of two
    equivalent Clifford+T circuits may not simplify to identity under full_reduce,
    leaving residual vertices (including non-Clifford phase gadgets). We treat all
    cases with internal vertices remaining as "unverified" rather than "failed",
    since residue from simplifier incompleteness is indistinguishable from true
    non-equivalence at this circuit size.
    """
    if original.qubits <= 10:
        try:
            # Exact equality check first (fast, works for small circuits)
            if original.qubits <= 6:
                try:
                    if original.verify_equality(optimized):
                        return "verified"
                except Exception:
                    pass
            # Unitary comparison (works up to ~10 qubits; 2^10=1024 matrix is fast)
            try:
                import numpy as np
                u = original.to_matrix()
                v = optimized.to_matrix()
                # Check exact equality first
                if np.allclose(u, v, atol=1e-8):
                    return "verified"
                # Check equality up to global phase: |tr(U†V)| = dim(U) iff V = e^{iθ}U
                udagv = np.conj(u).T @ v
                trace_val = np.trace(udagv)
                dim = u.shape[0]
                if abs(abs(trace_val) - dim) < 1e-6 * dim:
                    return "verified"
                return "failed"
            except Exception:
                return "unverified"
        except Exception:
            return "unverified"
    elif original.qubits <= 20:
        # Randomized state-vector sampling for 11-20 qubit circuits.
        # Full unitary comparison is too slow (9s+ for 11q), so we verify
        # by checking that k=5 random states map to matching outputs.
        # False-positive probability ≈ 1/2^(n*k) — negligible even for n=20.
        # State vector size: 2^20 = 1M elements × 16 bytes = 16MB (fast).
        return _statevec_equiv(original, optimized, k=5)
    else:
        # ZX-calculus identity test for larger circuits
        try:
            g_orig = original.to_graph()
            g_opt_adj = optimized.adjoint().to_graph()

            # Compose: original followed by adjoint(optimized)
            g_composed = g_orig + g_opt_adj

            # Simplify as far as possible
            zx.simplify.full_reduce(g_composed)

            # Classify internal vertices (non-boundary)
            internal_vertices = [
                v for v in g_composed.vertices()
                if g_composed.type(v) != zx.VertexType.BOUNDARY
            ]

            if len(internal_vertices) == 0:
                # No internal structure — provably identity
                return "verified"
            else:
                # Internal vertices remain. full_reduce is incomplete for Clifford+T,
                # so we cannot distinguish simplifier incompleteness from true
                # non-equivalence. Report as unverified.
                return "unverified"

        except Exception:
            return "unverified"


def _is_clifford_phase(phase) -> bool:
    """Check if a phase is Clifford (multiple of π/2, i.e., 0, 1/2, 1, 3/2)."""
    clifford_phases = {Fraction(0), Fraction(1, 2), Fraction(1, 1), Fraction(3, 2)}
    if isinstance(phase, Fraction):
        # Normalize to [0, 2)
        normalized = phase % 2
        return normalized in clifford_phases
    return False


def run_pipeline(circ: zx.Circuit, beam_width: int = 5, max_beam_depth: int = 20,
                 verbose: bool = False,
                 grpo_policy: "GRPOZXPolicy | None" = None,
                 bench_seed: "int | None" = None) -> dict:
    """
    Run the full three-stage optimization pipeline.

    grpo_policy: if provided, runs GRPO-guided stochastic rollouts (Path D)
                 as an additional optimization path alongside Path A and B.
    bench_seed:  if provided, re-seeds random before Path D for reproducibility
                 within each benchmark while preserving cross-benchmark learning.

    Returns a dict with per-stage results and the final optimized circuit.
    """
    original_stats = count_gates(circ.to_basic_gates())
    original_t = original_stats["t_count"]

    result = {
        "original_t_count": original_t,
        "original_stats": original_stats,
        "stages": {},
        "success": True,
    }

    # ---- Stage 1: ZX Preconditioning ----
    try:
        candidates, s1_time = stage1_zx_precondition(
            circ, beam_width=beam_width, max_depth=max_beam_depth, verbose=verbose
        )

        if candidates:
            best_s1_circ, best_s1_t = candidates[0]
        else:
            # Fallback: just use basic_gates version
            best_s1_circ = circ.to_basic_gates()
            best_s1_t = original_t

        s1_stats = count_gates(best_s1_circ)
        result["stages"]["s1_zx_precondition"] = {
            "t_count": s1_stats["t_count"],
            "stats": s1_stats,
            "time_s": s1_time,
            "n_candidates": len(candidates),
            "delta_from_original": original_t - s1_stats["t_count"],
        }
    except Exception as e:
        best_s1_circ = circ.to_basic_gates()
        result["stages"]["s1_zx_precondition"] = {
            "t_count": original_t,
            "time_s": 0,
            "error": str(e),
            "delta_from_original": 0,
        }

    # ---- Stage 2: Phase-Polynomial Resynthesis ----
    # Apply basic_optimization to Stage 1 output before TODD, then run TODD.
    # Try ALL Stage 1 candidates through Stage 2 and take the minimum T-count —
    # different Stage 1 circuits (even with the same T-count) can give very
    # different TODD results because TODD is sensitive to CNOT gate structure.
    s2_t0 = time.time()
    s2_circ = best_s1_circ  # fallback
    s2_best_t = float("inf")
    s2_improvements = []
    s1_t = result["stages"]["s1_zx_precondition"]["t_count"]

    # Collect all unique Stage 1 candidate circuits (up to top-5 by T-count)
    try:
        all_s1_candidates = candidates[:min(5, len(candidates))]
    except Exception:
        all_s1_candidates = [(best_s1_circ, s1_t)]

    for s1_cand, s1_cand_t in all_s1_candidates:
        try:
            c_in = zx.optimize.basic_optimization(s1_cand.to_basic_gates())
            c_out, _, impr = stage2_phase_poly_resynthesis(c_in, verbose=False)
            c_out3 = zx.optimize.basic_optimization(c_out.to_basic_gates())
            t_out = zx.tcount(c_out3)
            if t_out < s2_best_t:
                s2_best_t = t_out
                s2_circ = c_out3
                s2_improvements = impr
        except Exception:
            continue

    if s2_best_t == float("inf"):
        # All candidates failed; use fallback
        try:
            s2_circ = zx.optimize.basic_optimization(best_s1_circ.to_basic_gates())
        except Exception:
            s2_circ = best_s1_circ
        s2_best_t = count_gates(s2_circ)["t_count"]

    s2_stats = count_gates(s2_circ)
    s2_time = time.time() - s2_t0
    result["stages"]["s2_phase_poly"] = {
        "t_count": s2_stats["t_count"],
        "stats": s2_stats,
        "time_s": s2_time,
        "n_segments_optimized": len(s2_improvements),
        "segment_details": s2_improvements,
        "delta_from_s1": s1_t - s2_stats["t_count"],
    }

    # ---- Stage 3: Local Cleanup ----
    # Note: basic_opt is already applied at the end of Stage 2's candidate loop.
    # Stage 3 applies it one more time for any remaining cleanup.
    try:
        s3_circ, s3_time = stage3_local_cleanup(s2_circ, verbose=verbose)

        s3_stats = count_gates(s3_circ)
        s2_t = result["stages"]["s2_phase_poly"]["t_count"]
        result["stages"]["s3_cleanup"] = {
            "t_count": s3_stats["t_count"],
            "stats": s3_stats,
            "time_s": s3_time,
            "delta_from_s2": s2_t - s3_stats["t_count"],
        }
    except Exception as e:
        s3_circ = s2_circ
        result["stages"]["s3_cleanup"] = {
            "t_count": count_gates(s3_circ)["t_count"],
            "time_s": 0,
            "error": str(e),
            "delta_from_s2": 0,
        }

    # ---- Path B: basic_opt → TODD on ORIGINAL circuit (TODD-first order) ----
    # On some circuit families (especially toffoli_interleaved), ZX beam search
    # destroys the phase-polynomial structure that TODD needs to find reductions.
    # We apply our own basic_opt → TODD → basic_opt pipeline to the ORIGINAL
    # circuit as a parallel path. This is NOT full_optimize — we use the same
    # TODD (todd_simp) implementation as our Path A Stage 2.
    # We independently verify this path and take the minimum T-count.
    t_pathb_start = time.time()
    pathb_circ = None
    pathb_t = float("inf")
    try:
        cpb = zx.optimize.basic_optimization(circ.to_basic_gates())
        cpb_s2, _, _ = stage2_phase_poly_resynthesis(cpb, verbose=False)
        cpb_final = zx.optimize.basic_optimization(cpb_s2.to_basic_gates())
        pathb_t_val = zx.tcount(cpb_final)
        pathb_circ = cpb_final
        pathb_t = pathb_t_val
        result["stages"]["path_b_todd_first"] = {
            "t_count": pathb_t_val,
            "time_s": time.time() - t_pathb_start,
            "delta_from_original": original_t - pathb_t_val,
        }
    except Exception as e:
        result["stages"]["path_b_todd_first"] = {
            "t_count": None, "time_s": 0, "error": str(e),
        }

    # ---- Path F: ZX beam output → phase_block_optimize → basic_opt ----
    # Applies PyZX's phase_block_optimize (cross-HAD TODD, uses Optimizer.parse_circuit())
    # to ALL top-k Stage 1 ZX-simplified candidates. Unlike full_optimize (which applies
    # pbo to basic_opt(original)), we give phase_block_opt DIFFERENT starting circuits —
    # ones that have been ZX-preconditioned by our beam search. Different Stage 1 candidates
    # with similar T-counts but different structures may give different pbo results.
    # Random state is snapshot/restored to prevent contamination of subsequent operations.
    t_pathf_start = time.time()
    pathf_circ = None
    pathf_t = float("inf")
    # Snapshot random state before Path F to prevent contamination of later paths
    _rng_state_np = np.random.get_state()
    _rng_state_py = random.getstate()
    try:
        for s1_cand, _ in all_s1_candidates:
            try:
                c_pbo_out = zx.optimize.phase_block_optimize(s1_cand)
                c_pbo_final = zx.optimize.basic_optimization(c_pbo_out.to_basic_gates())
                pathf_t_val = zx.tcount(c_pbo_final)
                if pathf_t_val < pathf_t:
                    pathf_t = pathf_t_val
                    pathf_circ = c_pbo_final
            except Exception:
                continue
        result["stages"]["path_f_pbo"] = {
            "t_count": pathf_t if pathf_circ is not None else None,
            "time_s": time.time() - t_pathf_start,
            "delta_from_original": original_t - pathf_t if pathf_circ is not None else 0,
        }
    except Exception as e:
        result["stages"]["path_f_pbo"] = {
            "t_count": None, "time_s": 0, "error": str(e),
        }
    finally:
        # Restore random state so subsequent paths (E, D) are unaffected by Path F
        np.random.set_state(_rng_state_np)
        random.setstate(_rng_state_py)

    # Path E (ZX gadget post-processing) removed in iter19: analysis showed it was
    # redundant — all wins where path_e contributed were also achieved by other paths
    # (Stage 1/2, Path B, Path D, Path G). Removing it frees ~40s of time budget.
    pathe_circ = None
    pathe_t = float("inf")

    # ---- Path D: GRPO-guided stochastic ZX rollouts ----
    # Generates K=4 stochastic ZX rewrite rollouts guided by the learned GRPO policy.
    # The policy adapts across benchmarks: rules that led to better T-count reductions
    # get higher probability in future benchmarks. Each rollout applies the policy-sampled
    # rules, extracts a circuit, and runs basic_opt → TODD → basic_opt.
    # Path D is an independent exploration path that can find reductions missed by
    # the deterministic beam search (Path A) and the original-circuit TODD (Path B).
    pathd_circ = None
    pathd_t = float("inf")
    if grpo_policy is not None:
        try:
            # Re-seed for this benchmark's GRPO rollouts.
            # The policy's learned logits are deterministic (not random), so policy
            # learning from earlier benchmarks still affects these rollouts even though
            # the per-benchmark seed makes sampling reproducible across reruns.
            if bench_seed is not None:
                np.random.seed(bench_seed)
                random.seed(bench_seed)
            grpo_circ, grpo_t, _ = grpo_zx_search(
                circ, grpo_policy, K=4, max_depth=15, verbose=verbose
            )
            if grpo_circ is not None:
                # Pre-verify GRPO circuit before using it in final selection.
                # GRPO's stochastic rollouts can occasionally produce non-equivalent
                # circuits if ZX rewrite rules are applied in unexpected combinations.
                # We catch this here rather than letting the final verify fail.
                grpo_pre_verified = verify_equivalence(circ, grpo_circ)
                if grpo_pre_verified == "verified":
                    pathd_circ = grpo_circ
                    pathd_t = grpo_t
                elif grpo_pre_verified == "unverified":
                    pathd_circ = None  # Conservative: don't trust unverified GRPO circuit
                    pathd_t = float("inf")
                # "failed" → don't use (pathd_circ stays None)
            result["stages"]["path_d_grpo"] = {
                "t_count": pathd_t if pathd_circ is not None else None,
                "time_s": 0,  # included in rollout time
                "delta_from_original": original_t - pathd_t if pathd_circ is not None else 0,
            }
        except Exception as e:
            result["stages"]["path_d_grpo"] = {
                "t_count": None, "time_s": 0, "error": str(e),
            }

    # ---- Path G: Path B output → phase_block_optimize → basic_opt ----
    # Apply pbo to the TODD-first output (Path B). TODD rearranges T-gate structure
    # within Hadamard-free blocks; pbo may then find additional cross-HAD T reductions
    # that aren't visible in the original circuit. This combines TODD preprocessing
    # with pbo's cross-HAD capabilities, potentially beating either alone.
    # Uses only the Path B circuit (single pbo call — minimal extra time budget).
    pathg_circ = None
    pathg_t = float("inf")
    if pathb_circ is not None:
        t_pathg_start = time.time()
        _rng_state_np_g = np.random.get_state()
        _rng_state_py_g = random.getstate()
        try:
            c_pbog_out = zx.optimize.phase_block_optimize(pathb_circ)
            c_pbog_final = zx.optimize.basic_optimization(c_pbog_out.to_basic_gates())
            pathg_t_val = zx.tcount(c_pbog_final)
            pathg_circ = c_pbog_final
            pathg_t = pathg_t_val
            result["stages"]["path_g_todd_pbo"] = {
                "t_count": pathg_t_val,
                "time_s": time.time() - t_pathg_start,
                "delta_from_original": original_t - pathg_t_val,
            }
        except Exception as e:
            result["stages"]["path_g_todd_pbo"] = {"t_count": None, "time_s": 0, "error": str(e)}
        finally:
            np.random.set_state(_rng_state_np_g)
            random.setstate(_rng_state_py_g)

    # ---- Path I: GRPO best rollout circuit → phase_block_optimize → basic_opt ----
    # GRPO's stochastic exploration sometimes finds circuit structures different from
    # deterministic beam search. Applying pbo to the GRPO-found circuit may exploit
    # cross-HAD phase structure that TODD alone (used inside GRPO rollouts) cannot find.
    # This path only runs if Path D (GRPO) produced a result.
    pathi_circ = None
    pathi_t = float("inf")
    if pathd_circ is not None:
        t_pathi_start = time.time()
        _rng_state_np_i = np.random.get_state()
        _rng_state_py_i = random.getstate()
        try:
            c_pboi_out = zx.optimize.phase_block_optimize(pathd_circ)
            c_pboi_final = zx.optimize.basic_optimization(c_pboi_out.to_basic_gates())
            pathi_t_val = zx.tcount(c_pboi_final)
            pathi_circ = c_pboi_final
            pathi_t = pathi_t_val
            result["stages"]["path_i_grpo_pbo"] = {
                "t_count": pathi_t_val,
                "time_s": time.time() - t_pathi_start,
                "delta_from_original": original_t - pathi_t_val,
            }
        except Exception as e:
            result["stages"]["path_i_grpo_pbo"] = {"t_count": None, "time_s": 0, "error": str(e)}
        finally:
            np.random.set_state(_rng_state_np_i)
            random.setstate(_rng_state_py_i)

    # ---- Final result: take minimum across all paths ----
    # Paths: A (ZX beam + TODD), B (TODD on original), D (GRPO rollouts),
    #        F (ZX beam + phase_block_opt), G (TODD on original → pbo),
    #        I (GRPO best rollout → pbo)
    s3_t = count_gates(s3_circ)["t_count"]
    best_t = s3_t
    final_circ = s3_circ
    if pathb_circ is not None and pathb_t < best_t:
        best_t = pathb_t
        final_circ = pathb_circ
    if pathf_circ is not None and pathf_t < best_t:
        # Pre-verify Path F (phase_block_optimize can produce global-phase equivalences)
        if verify_equivalence(circ, pathf_circ) == "verified":
            best_t = pathf_t
            final_circ = pathf_circ
    if pathg_circ is not None and pathg_t < best_t:
        if verify_equivalence(circ, pathg_circ) == "verified":
            best_t = pathg_t
            final_circ = pathg_circ
    if pathd_circ is not None and pathd_t < best_t:
        best_t = pathd_t
        final_circ = pathd_circ
    if pathi_circ is not None and pathi_t < best_t:
        if verify_equivalence(circ, pathi_circ) == "verified":
            best_t = pathi_t
            final_circ = pathi_circ
    s3_circ = final_circ

    # ---- Final result ----
    final_stats = count_gates(s3_circ)
    total_time = sum(
        result["stages"][s].get("time_s", 0)
        for s in ["s1_zx_precondition", "s2_phase_poly", "s3_cleanup"]
    )

    # Verify equivalence
    verified = verify_equivalence(circ, s3_circ)

    result["final"] = {
        "t_count": final_stats["t_count"],
        "stats": final_stats,
        "total_time_s": total_time,
        "delta_from_original": original_t - final_stats["t_count"],
        "verified": verified,
    }

    if verified == "failed":
        # Equivalence check failed — result is INVALID
        result["success"] = False
        result["final"]["INVALID"] = True
        print("    *** EQUIVALENCE CHECK FAILED — result discarded ***")
    elif verified == "unverified":
        # Could not verify — mark as unverified, still counts as a result
        # but must be reported separately in summary stats
        result["success"] = True
        result["final"]["unverified"] = True
        print("    *** EQUIVALENCE UNVERIFIED (circuit too large or error) ***")

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run T-count optimization pipeline")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-beam-depth", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load baselines
    baselines_path = benchmarks_dir / "baselines.json"
    if not baselines_path.exists():
        print(f"ERROR: {baselines_path} not found. Run prepare.py first.")
        return

    with open(baselines_path) as f:
        baselines = json.load(f)

    print(f"Loaded {len(baselines)} benchmarks from {baselines_path}")
    print(f"Beam width: {args.beam_width}, Max beam depth: {args.max_beam_depth}")
    # GRPO pre-training on SYNTHETIC circuits (NOT the test benchmarks).
    # This ensures clean evaluation: the policy is trained on a separate dataset,
    # then frozen before evaluating on the 50 test benchmarks (no test-set contamination).
    #
    # Warm-up circuits are generated with fixed seeds using PyZX's CNOT_HAD_PHASE_circuit,
    # covering the same qubit/depth range as the test benchmarks.
    grpo_policy = GRPOZXPolicy(n_rules=len(ZX_REWRITE_RULES), lr=0.05, temperature=1.0)

    WARMUP_CONFIGS = [  # (qubits, depth) — mirrors test benchmark size distribution
        (4, 60), (5, 80), (6, 100), (6, 150), (8, 100),
        (8, 160), (10, 120), (10, 200), (4, 100), (5, 120),
        (6, 80), (8, 200), (10, 160), (4, 60), (5, 80),
        (6, 120), (8, 100), (10, 120), (4, 80), (6, 100),
    ]
    print(f"Pre-training GRPO policy on {len(WARMUP_CONFIGS)} synthetic warm-up circuits (K=4)...")
    for w_i, (n_q, n_depth) in enumerate(WARMUP_CONFIGS):
        try:
            warm_circ = zx.generate.CNOT_HAD_PHASE_circuit(
                qubits=n_q, depth=n_depth, p_had=0.2, p_t=0.4,
                seed=RANDOM_SEED + 20000 + w_i
            )
            np.random.seed(RANDOM_SEED + 20000 + w_i)
            random.seed(RANDOM_SEED + 20000 + w_i)
            grpo_zx_search(warm_circ, grpo_policy, K=4, max_depth=15, verbose=False)  # return value unused in warmup
        except Exception:
            pass

    print(f"Warm-up done. Learned policy probabilities:")
    probs_warmup = grpo_policy.get_probs()
    for (name, _), p in zip(ZX_REWRITE_RULES, probs_warmup):
        print(f"  {name}: {p:.3f}")
    print("Freezing policy for clean evaluation (no further updates on test benchmarks).")
    grpo_policy.freeze()

    # Reset main random seeds before evaluation to isolate warm-up effects
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print()

    # Save configuration
    config = {
        "beam_width": args.beam_width,
        "max_beam_depth": args.max_beam_depth,
        "pyzx_version": zx.__version__,
        "n_benchmarks": len(baselines),
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run pipeline on each benchmark
    all_results = {}
    summary_stats = {
        "total": 0,
        "successful": 0,
        "verified": 0,
        "unverified": 0,
        "verification_failures": 0,
        "pipeline_errors": 0,
        # Win/loss/tie counted ONLY over verified results
        "wins_vs_full_reduce": 0,
        "wins_vs_full_optimize": 0,
        "ties_vs_full_optimize": 0,
        "losses_vs_full_optimize": 0,
        # Unverified results that would be wins/ties/losses if trusted
        "unverified_wins_vs_full_optimize": 0,
        "unverified_ties_vs_full_optimize": 0,
        "unverified_losses_vs_full_optimize": 0,
    }

    for i, (name, entry) in enumerate(baselines.items()):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(baselines)}] {name}")
        print(f"{'='*60}")

        summary_stats["total"] += 1

        # Load circuit from QASM
        label = name.split("/")[1] if "/" in name else name
        qasm_path = benchmarks_dir / "circuits" / f"{label}.qasm"

        if not qasm_path.exists():
            print(f"  WARNING: QASM file not found: {qasm_path}")
            summary_stats["pipeline_errors"] += 1
            continue

        try:
            circ = zx.Circuit.from_qasm_file(str(qasm_path))
        except Exception as e:
            print(f"  ERROR loading circuit: {e}")
            summary_stats["pipeline_errors"] += 1
            continue

        orig_t = entry["baselines"]["original"]["t_count"]
        print(f"  Original T-count: {orig_t}, Qubits: {entry['qubits']}")

        # Print baseline T-counts for reference
        for bname in ["full_reduce", "full_optimize", "phase_block"]:
            bdata = entry["baselines"].get(bname, {})
            if bdata.get("success"):
                print(f"  Baseline {bname}: T={bdata['t_count']}")

        # Run our pipeline
        try:
            result = run_pipeline(
                circ,
                beam_width=args.beam_width,
                max_beam_depth=args.max_beam_depth,
                verbose=args.verbose,
                grpo_policy=grpo_policy,
                bench_seed=RANDOM_SEED + i,
            )
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            traceback.print_exc()
            summary_stats["pipeline_errors"] += 1
            all_results[name] = {"success": False, "error": str(e)}
            continue

        # Compare with baselines
        our_t = result["final"]["t_count"]
        fr_t = entry["baselines"].get("full_reduce", {}).get("t_count", float("inf"))
        fo_t = entry["baselines"].get("full_optimize", {}).get("t_count", float("inf"))

        print(f"\n  RESULT: T={our_t} (Δ={orig_t - our_t:+d} from original)")
        print(f"  vs full_reduce (T={fr_t}):  {'WIN' if our_t < fr_t else 'TIE' if our_t == fr_t else 'LOSS'}")
        print(f"  vs full_optimize (T={fo_t}): {'WIN' if our_t < fo_t else 'TIE' if our_t == fo_t else 'LOSS'}")

        if result["final"].get("verified") is not None:
            print(f"  Verified: {result['final']['verified']}")

        # Per-stage breakdown
        for stage_name, stage_data in result["stages"].items():
            delta_key = [k for k in stage_data if k.startswith("delta_")]
            delta_val = stage_data[delta_key[0]] if delta_key else 0
            print(f"  {stage_name}: T={stage_data['t_count']}, "
                  f"Δ={delta_val:+d}, time={stage_data.get('time_s', 0):.2f}s")

        # Update summary — count EVERY benchmark, not just successful ones
        if result["success"]:
            summary_stats["successful"] += 1
            v = result["final"].get("verified", "unverified")
            is_verified = (v == "verified")

            if v == "verified":
                summary_stats["verified"] += 1
            elif v == "unverified":
                summary_stats["unverified"] += 1

            # Win/loss/tie comparison — ONLY verified results go into main counts
            if entry["baselines"].get("full_optimize", {}).get("success"):
                if our_t < fo_t:
                    if is_verified:
                        summary_stats["wins_vs_full_optimize"] += 1
                    else:
                        summary_stats["unverified_wins_vs_full_optimize"] += 1
                elif our_t == fo_t:
                    if is_verified:
                        summary_stats["ties_vs_full_optimize"] += 1
                    else:
                        summary_stats["unverified_ties_vs_full_optimize"] += 1
                else:
                    if is_verified:
                        summary_stats["losses_vs_full_optimize"] += 1
                    else:
                        summary_stats["unverified_losses_vs_full_optimize"] += 1

            if entry["baselines"].get("full_reduce", {}).get("success"):
                if our_t < fr_t and is_verified:
                    summary_stats["wins_vs_full_reduce"] += 1
        else:
            if result["final"].get("verified") == "failed":
                summary_stats["verification_failures"] += 1
            else:
                summary_stats["pipeline_errors"] += 1

        # Store result alongside baseline data for comparison
        all_results[name] = {
            "pipeline_result": result,
            "baselines": entry["baselines"],
            "qubits": entry["qubits"],
            "family": entry["family"],
        }

    # Save all results
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary — report ALL benchmarks with clear verified/unverified split
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_total = summary_stats["total"]
    n_verified = summary_stats["verified"]
    n_unverified = summary_stats["unverified"]
    print(f"Total benchmarks:           {n_total}")
    print(f"  Successful runs:          {summary_stats['successful']}")
    print(f"    Verified equivalent:    {n_verified}")
    print(f"    Unverified (>6q/error): {n_unverified}")
    print(f"  Verification failures:    {summary_stats['verification_failures']}")
    print(f"  Pipeline errors:          {summary_stats['pipeline_errors']}")
    print()
    test_metric = summary_stats["wins_vs_full_optimize"] / n_total if n_total > 0 else 0.0
    print(f"TEST_METRIC = {test_metric:.4f} ({summary_stats['wins_vs_full_optimize']}/{n_total} benchmarks)")
    print()
    print(f"Win/Tie/Loss vs full_optimize — VERIFIED ONLY (denominator = {n_verified}):")
    print(f"  Wins:   {summary_stats['wins_vs_full_optimize']}/{n_verified}")
    print(f"  Ties:   {summary_stats['ties_vs_full_optimize']}/{n_verified}")
    print(f"  Losses: {summary_stats['losses_vs_full_optimize']}/{n_verified}")
    print(f"\nWins vs full_reduce (verified only): "
          f"{summary_stats['wins_vs_full_reduce']}/{n_verified}")
    if n_unverified > 0:
        print(f"\nUnverified results (NOT included in counts above):")
        print(f"  Would-be wins:   {summary_stats['unverified_wins_vs_full_optimize']}/{n_unverified}")
        print(f"  Would-be ties:   {summary_stats['unverified_ties_vs_full_optimize']}/{n_unverified}")
        print(f"  Would-be losses: {summary_stats['unverified_losses_vs_full_optimize']}/{n_unverified}")
    print()

    # Print learned GRPO policy
    probs = grpo_policy.get_probs()
    print("Learned GRPO policy (rule probabilities after all benchmarks):")
    for (name, _), p in zip(ZX_REWRITE_RULES, probs):
        print(f"  {name}: {p:.3f}")
    print()

    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Results saved to {results_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
