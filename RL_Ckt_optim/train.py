"""
Quantum Circuit Compilation via Variational Optimization
=========================================================
Compiles target unitaries into minimal-CNOT circuits by optimizing
rotation angles in parameterized templates.

TEST_ACC = fraction of held-out unitaries where compiled circuit uses
fewer 2-qubit gates than Qiskit opt-3, with fidelity >= 0.999.
"""

import os, time, json, random, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIDELITY_THR = 0.999
MAX_RUN_TIME = 295
SEED = 42

_GRPO_POLICY = None  # global GRPO policy, set during training

# ─── CX embedding cache ───
_cx_cache = {}

def _embed_cx(ctrl, targ, n_qubits):
    """Embed CX(ctrl, targ) into full 2^n Hilbert space. Cached."""
    key = (ctrl, targ, n_qubits)
    if key in _cx_cache:
        return _cx_cache[key]
    dim = 2 ** n_qubits
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            bits_i = [(i >> (n_qubits-1-k)) & 1 for k in range(n_qubits)]
            bits_j = [(j >> (n_qubits-1-k)) & 1 for k in range(n_qubits)]
            if all(bits_i[k] == bits_j[k] for k in range(n_qubits) if k != ctrl and k != targ):
                r = bits_i[ctrl]*2 + bits_i[targ]
                c = bits_j[ctrl]*2 + bits_j[targ]
                mat[i,j] = CX[r,c]
    t = torch.tensor(mat, dtype=torch.complex128)
    _cx_cache[key] = t
    return t


# ─── Differentiable U3 gate (ZYZ decomposition) ───

def _u3(a, b, g):
    """RZ(a) @ RY(b) @ RZ(g) → 2x2 complex tensor, differentiable."""
    ca, sa = torch.cos(a/2), torch.sin(a/2)
    cb, sb = torch.cos(b/2), torch.sin(b/2)
    cg, sg = torch.cos(g/2), torch.sin(g/2)
    ea_n = torch.complex(ca, -sa)
    ea_p = torch.complex(ca,  sa)
    eg_n = torch.complex(cg, -sg)
    eg_p = torch.complex(cg,  sg)
    return torch.stack([
        torch.stack([ea_n * cb * eg_n, -ea_n * sb * eg_p]),
        torch.stack([ea_p * sb * eg_n,  ea_p * cb * eg_p]),
    ])


def _sq_layer(params, nq):
    """Kronecker product of nq single-qubit U3 gates. params: (nq, 3)."""
    m = _u3(params[0, 0], params[0, 1], params[0, 2])
    for q in range(1, nq):
        m = torch.kron(m, _u3(params[q, 0], params[q, 1], params[q, 2]))
    return m


def build_unitary(params, nq, cx_pairs):
    """
    Build compiled unitary from parameterized circuit.
    params: (n_cx + 1, nq, 3) — angles for single-qubit layers.
    cx_pairs: list of (ctrl, targ) for CX gates between layers.
    """
    dim = 2 ** nq
    U = torch.eye(dim, dtype=torch.complex128)
    idx = 0
    for c, t in cx_pairs:
        U = _sq_layer(params[idx], nq) @ U
        idx += 1
        U = _embed_cx(c, t, nq) @ U
    U = _sq_layer(params[idx], nq) @ U
    return U


def proc_fidelity(Ut, Uc):
    """Process fidelity |Tr(Ut† Uc)|² / d²."""
    d = Ut.shape[0]
    return torch.abs(torch.trace(Ut.conj().T @ Uc)) ** 2 / d ** 2


# ─── CX placement patterns ───

CX_PATTERNS_3Q = [
    [(0,1), (1,2), (0,2)],   # triangle forward
    [(0,1), (0,2), (1,2)],   # fan-out from 0
]

CX_PATTERNS_4Q = [
    [(0,1), (1,2), (2,3), (0,2), (1,3), (0,3)],
    [(0,1), (2,3), (1,2), (0,3)],
]

CX_PATTERNS_5Q = [
    [(0,1), (1,2), (2,3), (3,4), (0,2), (1,3), (2,4), (0,3), (1,4), (0,4)],
]


def get_patterns(nq):
    if nq == 3:
        return CX_PATTERNS_3Q
    elif nq == 4:
        return CX_PATTERNS_4Q
    elif nq == 5:
        return CX_PATTERNS_5Q
    return [[(i, j) for i in range(nq) for j in range(nq) if i != j]]


def cem_compile(U_target_t, nq, n_cx, pattern, n_samples=80, n_elite=15,
                n_iters=15, adam_steps=60, lr=0.1):
    """
    Cross-Entropy Method (CEM) + Adam fine-tuning for circuit compilation.
    CEM: evolutionary RL method that maintains a Gaussian over parameters,
    samples, evaluates, and updates based on elite (top-performing) samples.
    Returns (fidelity, success, best_params).
    """
    cx_pairs = [pattern[i % len(pattern)] for i in range(n_cx)]
    n_layers = n_cx + 1
    n_params = n_layers * nq * 3

    # Initialize CEM distribution
    mean = torch.zeros(n_params, dtype=torch.float64)
    std = torch.ones(n_params, dtype=torch.float64) * 0.5

    best_fid_overall = 0.0
    best_params_overall = None

    for it in range(n_iters):
        # Sample from current distribution
        noise = torch.randn(n_samples, n_params, dtype=torch.float64)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise

        # Evaluate all samples (no gradient needed)
        fids = torch.zeros(n_samples)
        with torch.no_grad():
            for si in range(n_samples):
                params = samples[si].reshape(n_layers, nq, 3)
                U = build_unitary(params, nq, cx_pairs)
                fids[si] = proc_fidelity(U_target_t, U)

        # Elite selection
        elite_idx = torch.argsort(fids, descending=True)[:n_elite]
        elite = samples[elite_idx]

        # Update distribution
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp(min=0.01)

        best_fid_iter = fids[elite_idx[0]].item()
        if best_fid_iter > best_fid_overall:
            best_fid_overall = best_fid_iter
            best_params_overall = samples[elite_idx[0]].reshape(n_layers, nq, 3).clone()

        if best_fid_iter >= FIDELITY_THR:
            return best_fid_iter, True, best_params_overall.detach()

    # Phase 2: Adam fine-tuning on best CEM solution
    if best_params_overall is not None:
        params = best_params_overall.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        for step in range(adam_steps):
            opt.zero_grad()
            U = build_unitary(params, nq, cx_pairs)
            fid = proc_fidelity(U_target_t, U)
            fv = fid.item()
            if fv >= FIDELITY_THR:
                return fv, True, params.detach()
            (1.0 - fid).backward()
            opt.step()

        # L-BFGS polish
        if fv > 0.5:
            params_l = params.detach().clone().requires_grad_(True)
            def closure():
                opt_l.zero_grad()
                U = build_unitary(params_l, nq, cx_pairs)
                loss = 1.0 - proc_fidelity(U_target_t, U)
                loss.backward()
                return loss
            opt_l = torch.optim.LBFGS([params_l], lr=1.0, max_iter=20,
                                       history_size=10, line_search_fn="strong_wolfe")
            for _ in range(3):
                opt_l.step(closure)
                U = build_unitary(params_l, nq, cx_pairs)
                fv = proc_fidelity(U_target_t, U).item()
                if fv >= FIDELITY_THR:
                    return fv, True, params_l.detach()

        return fv, False, params.detach()

    return 0.0, False, None


# ─── GRPO: Group Relative Policy Optimization for circuit compilation ───

class CompilationPolicy(nn.Module):
    """MLP policy: maps target unitary → circuit rotation angles."""
    def __init__(self, state_dim, param_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, param_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(param_dim) - 0.5)

    def forward(self, x):
        mean = self.net(x)
        return mean

    def sample(self, x, n_samples=8):
        """Sample n_samples parameter vectors from the policy."""
        mean = self.forward(x)
        std = torch.exp(self.log_std).unsqueeze(0)
        noise = torch.randn(n_samples, mean.shape[-1])
        samples = mean.unsqueeze(0) + std * noise
        log_probs = -0.5 * ((samples - mean.unsqueeze(0)) / std) ** 2 - self.log_std.unsqueeze(0)
        log_probs = log_probs.sum(dim=-1)
        return samples, log_probs


def train_grpo_policy(benchmarks, nq, n_cx, pattern, n_epochs=3, group_size=8,
                      lr_policy=1e-3, adam_finetune_steps=30):
    """
    Train a GRPO policy for circuit compilation.
    GRPO: sample a group of circuits per unitary, compute group-relative advantages,
    update policy to favor high-reward (high-fidelity, low-gate) circuits.
    """
    cx_pairs = [pattern[i % len(pattern)] for i in range(n_cx)]
    n_layers = n_cx + 1
    param_dim = n_layers * nq * 3
    state_dim = 2 * (2**nq)**2  # flattened real+imag of target unitary

    policy = CompilationPolicy(state_dim, param_dim)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr_policy)

    # Get training unitaries
    train_keys = [k for k, v in benchmarks.items() if v["n_qubits"] == nq]
    random.shuffle(train_keys)
    train_keys = train_keys[:int(0.8 * len(train_keys))]

    for epoch in range(n_epochs):
        total_reward = 0.0
        n_samples = 0
        for key in train_keys:
            U_target = benchmarks[key]["unitary"]
            U_t = torch.tensor(U_target, dtype=torch.complex128)

            # State: flattened real+imag of target unitary
            state = torch.cat([
                torch.tensor(U_target.real.flatten(), dtype=torch.float32),
                torch.tensor(U_target.imag.flatten(), dtype=torch.float32),
            ])

            # Sample group of circuit parameterizations
            samples, log_probs = policy.sample(state, n_samples=group_size)

            # Evaluate each sample
            rewards = torch.zeros(group_size)
            with torch.no_grad():
                for gi in range(group_size):
                    params = samples[gi].reshape(n_layers, nq, 3).to(torch.float64)
                    U_compiled = build_unitary(params, nq, cx_pairs)
                    fid = proc_fidelity(U_t, U_compiled).item()
                    rewards[gi] = fid  # reward = fidelity

            # GRPO: group-relative advantage
            if rewards.std() > 1e-8:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantages = torch.zeros_like(rewards)

            # Policy gradient update
            policy_loss = -(advantages.detach() * log_probs).mean()
            policy_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_opt.step()

            total_reward += rewards.max().item()
            n_samples += 1

        avg_reward = total_reward / max(n_samples, 1)
        print(f"    GRPO epoch {epoch+1}/{n_epochs}: avg_best_fid={avg_reward:.4f}")

    return policy


def grpo_compile(policy, U_target_np, nq, n_cx, pattern,
                 group_size=16, adam_steps=50, lr=0.1):
    """Use trained GRPO policy to compile a unitary. Returns (fid, success, params)."""
    cx_pairs = [pattern[i % len(pattern)] for i in range(n_cx)]
    n_layers = n_cx + 1
    U_t = torch.tensor(U_target_np, dtype=torch.complex128)

    state = torch.cat([
        torch.tensor(U_target_np.real.flatten(), dtype=torch.float32),
        torch.tensor(U_target_np.imag.flatten(), dtype=torch.float32),
    ])

    # Sample from policy
    with torch.no_grad():
        samples, _ = policy.sample(state, n_samples=group_size)
        best_fid = 0.0
        best_params = None
        for gi in range(group_size):
            params = samples[gi].reshape(n_layers, nq, 3).to(torch.float64)
            U_c = build_unitary(params, nq, cx_pairs)
            fid = proc_fidelity(U_t, U_c).item()
            if fid > best_fid:
                best_fid = fid
                best_params = params

    if best_params is None:
        return 0.0, False, None

    # Adam fine-tuning
    params = best_params.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)
    for step in range(adam_steps):
        opt.zero_grad()
        U_c = build_unitary(params, nq, cx_pairs)
        fid = proc_fidelity(U_t, U_c)
        fv = fid.item()
        if fv >= FIDELITY_THR:
            return fv, True, params.detach()
        (1.0 - fid).backward()
        opt.step()

    # L-BFGS polish
    if fv > 0.5:
        params_l = params.detach().clone().requires_grad_(True)
        def closure():
            opt_l.zero_grad()
            U_c = build_unitary(params_l, nq, cx_pairs)
            loss = 1.0 - proc_fidelity(U_t, U_c)
            loss.backward()
            return loss
        opt_l = torch.optim.LBFGS([params_l], lr=1.0, max_iter=20,
                                   history_size=10, line_search_fn="strong_wolfe")
        for _ in range(3):
            opt_l.step(closure)
            U_c = build_unitary(params_l, nq, cx_pairs)
            fv = proc_fidelity(U_t, U_c).item()
            if fv >= FIDELITY_THR:
                return fv, True, params_l.detach()

    return fv, False, params.detach()


# ─── RL Agent: REINFORCE for CX pair selection ───

_RL_POLICY = None  # global RL policy, set during training

class RLCircuitPolicy(nn.Module):
    """MLP policy that autoregressively selects CX pairs.
    Input: flattened real+imag of residual unitary R.
    Output: logits over CX pair actions + STOP action.
    """
    def __init__(self, nq, hidden=128):
        super().__init__()
        self.nq = nq
        dim = 2 ** nq
        state_dim = 2 * dim * dim  # real + imag of residual
        # All directed CX pairs + STOP
        self.cx_actions = [(i, j) for i in range(nq) for j in range(nq) if i != j]
        n_actions = len(self.cx_actions) + 1  # +1 for STOP
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        """Returns action logits given state."""
        return self.net(state)

    def get_action(self, state, greedy=False):
        """Sample or greedily pick an action. Returns (action_idx, log_prob)."""
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        if greedy:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        log_prob = torch.log(probs[action] + 1e-10)
        return action, log_prob


def _residual_state(U_target_t, cx_pairs_so_far, nq):
    """Compute residual unitary R = U_target^dag @ U_cx_sequence and flatten to state vector.
    U_cx_sequence is just the CX gates applied so far (no rotation optimization yet)."""
    dim = 2 ** nq
    U_cx = torch.eye(dim, dtype=torch.complex128)
    for c, t in cx_pairs_so_far:
        U_cx = _embed_cx(c, t, nq) @ U_cx
    R = U_target_t.conj().T @ U_cx
    state = torch.cat([R.real.flatten().float(), R.imag.flatten().float()])
    return state


def _optimize_angles(U_target_t, nq, cx_pairs, adam_steps=60, lr=0.1):
    """Optimize rotation angles for a given CX sequence. Returns (fidelity, params)."""
    n_layers = len(cx_pairs) + 1
    best_fid = 0.0
    best_params = None

    for trial in range(2):
        params = torch.randn(n_layers, nq, 3, dtype=torch.float64) * 0.3
        params = params.detach().requires_grad_(True)

        opt = torch.optim.Adam([params], lr=lr)
        fv = 0.0
        for step in range(adam_steps):
            opt.zero_grad()
            Uc = build_unitary(params, nq, cx_pairs)
            fid = proc_fidelity(U_target_t, Uc)
            fv = fid.item()
            if fv >= FIDELITY_THR:
                return fv, params.detach()
            (1.0 - fid).backward()
            opt.step()
            # Early exit for hopeless trials
            if step == 20 and fv < 0.05:
                break

        # L-BFGS polish if promising
        if fv > 0.5:
            params_l = params.detach().clone().requires_grad_(True)
            def closure():
                opt_l.zero_grad()
                Uc = build_unitary(params_l, nq, cx_pairs)
                loss = 1.0 - proc_fidelity(U_target_t, Uc)
                loss.backward()
                return loss
            opt_l = torch.optim.LBFGS([params_l], lr=1.0, max_iter=20,
                                       history_size=10, line_search_fn="strong_wolfe")
            for _ in range(3):
                opt_l.step(closure)
                Uc = build_unitary(params_l, nq, cx_pairs)
                fv = proc_fidelity(U_target_t, Uc).item()
                if fv >= FIDELITY_THR:
                    return fv, params_l.detach()
            if fv > best_fid:
                best_fid = fv
                best_params = params_l.detach()
        else:
            if fv > best_fid:
                best_fid = fv
                best_params = params.detach()

    return best_fid, best_params


def train_rl_policy(benchmarks, nq, n_epochs=2, max_cx=12, lr_policy=3e-3):
    """
    Train RL policy using REINFORCE with baseline.
    Each episode: agent picks CX pairs autoregressively, then angles are optimized.
    Reward: 10*(1 - n_cx/max_cx) if fidelity >= 0.999, else -5.
    """
    policy = RLCircuitPolicy(nq, hidden=128)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr_policy)

    train_keys = [k for k, v in benchmarks.items() if v["n_qubits"] == nq]
    random.shuffle(train_keys)
    n_train = min(int(0.8 * len(train_keys)), 80)
    train_keys = train_keys[:n_train]

    baseline_val = 0.0  # running baseline for variance reduction
    baseline_alpha = 0.1

    print(f"  RL training: {n_train} unitaries, {n_epochs} epochs, max_cx={max_cx}")
    t_start = time.time()

    for epoch in range(n_epochs):
        epoch_rewards = []
        epoch_successes = 0

        for ki, key in enumerate(train_keys):
            # Time guard: 60s total RL training budget
            if time.time() - t_start > 55:
                print(f"    RL time limit at epoch {epoch+1}, unitary {ki}")
                break

            info = benchmarks[key]
            U_target = info["unitary"]
            U_t = torch.tensor(U_target, dtype=torch.complex128)

            # Get baseline CX count for reward scaling
            bl = info.get("baselines", {}).get("ibm_opt3", {})
            bl_2q = bl.get("two_qubit_gates", max_cx) if bl.get("success") else max_cx

            # Episode: autoregressively select CX pairs
            cx_pairs = []
            log_probs = []
            state = _residual_state(U_t, cx_pairs, nq)

            for step in range(max_cx):
                action, log_prob = policy.get_action(state)
                log_probs.append(log_prob)

                # STOP action
                if action == policy.n_actions - 1:
                    break

                cx_pair = policy.cx_actions[action]
                cx_pairs.append(cx_pair)
                state = _residual_state(U_t, cx_pairs, nq)

            n_cx_used = len(cx_pairs)

            # Optimize rotation angles (reduced steps during training for speed)
            if n_cx_used == 0:
                fid, _ = _optimize_angles(U_t, nq, [], adam_steps=20, lr=0.15)
            else:
                fid, _ = _optimize_angles(U_t, nq, cx_pairs, adam_steps=25, lr=0.15)

            # Compute reward
            if fid >= FIDELITY_THR:
                reward = 10.0 * (1.0 - n_cx_used / max_cx)
                epoch_successes += 1
            else:
                reward = -5.0

            epoch_rewards.append(reward)

            # REINFORCE update with baseline
            advantage = reward - baseline_val
            baseline_val = baseline_val + baseline_alpha * (reward - baseline_val)

            if len(log_probs) > 0:
                policy_loss = -advantage * torch.stack(log_probs).sum()
                policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                policy_opt.step()

        avg_r = np.mean(epoch_rewards) if epoch_rewards else 0.0
        print(f"    RL epoch {epoch+1}/{n_epochs}: avg_reward={avg_r:.2f}, "
              f"successes={epoch_successes}/{len(epoch_rewards)}, "
              f"time={time.time()-t_start:.1f}s")

        if time.time() - t_start > 55:
            break

    print(f"  RL training done in {time.time()-t_start:.1f}s")
    return policy


def rl_compile(policy, U_target_np, nq, max_cx=12, n_rollouts=5):
    """
    Use trained RL policy to compile a unitary.
    Runs multiple rollouts and picks the best (fewest CX with fidelity >= threshold).
    Returns (n_cx, fidelity, success).
    """
    U_t = torch.tensor(U_target_np, dtype=torch.complex128)

    best_cx = max_cx + 1
    best_fid = 0.0
    best_success = False

    for rollout in range(n_rollouts):
        cx_pairs = []
        state = _residual_state(U_t, cx_pairs, nq)

        for step in range(max_cx):
            # Use greedy on first rollout, sample on rest
            greedy = (rollout == 0)
            with torch.no_grad():
                action, _ = policy.get_action(state, greedy=greedy)

            if action == policy.n_actions - 1:  # STOP
                break

            cx_pair = policy.cx_actions[action]
            cx_pairs.append(cx_pair)
            state = _residual_state(U_t, cx_pairs, nq)

        n_cx_used = len(cx_pairs)
        fid, params = _optimize_angles(U_t, nq, cx_pairs, adam_steps=60, lr=0.1)

        if fid >= FIDELITY_THR and n_cx_used < best_cx:
            best_cx = n_cx_used
            best_fid = fid
            best_success = True
        elif not best_success and fid > best_fid:
            best_cx = n_cx_used
            best_fid = fid

    if best_success:
        return best_cx, best_fid, True
    return best_cx, best_fid, False


def try_compile(U_target_t, nq, n_cx, patterns, n_trials=3, n_steps=200, lr=0.1,
                init_params=None):
    """
    Try to compile U_target with exactly n_cx CX gates.
    Returns (fidelity, success, best_params).
    """
    best_fid = 0.0
    best_params = None

    for pat in patterns:
        cx_pairs = [pat[i % len(pat)] for i in range(n_cx)]
        n_layers = n_cx + 1

        for trial in range(n_trials):
            if init_params is not None and trial == 0 and init_params.shape[0] >= n_layers:
                # Warm start: use parameters from a higher CX solution
                params = init_params[:n_layers].clone().detach().requires_grad_(True)
            else:
                params = torch.randn(n_layers, nq, 3, dtype=torch.float64) * 0.3
                params = params.detach().requires_grad_(True)

            # Phase 1: Adam for coarse optimization
            adam_steps = min(n_steps, 100)
            opt = torch.optim.Adam([params], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=adam_steps, eta_min=lr / 20
            )

            for step in range(adam_steps):
                opt.zero_grad()
                Uc = build_unitary(params, nq, cx_pairs)
                fid = proc_fidelity(U_target_t, Uc)
                fv = fid.item()

                if fv >= FIDELITY_THR:
                    return fv, True, params.detach()

                (1.0 - fid).backward()
                opt.step()
                sched.step()

                if step == 30 and fv < 0.05:
                    break
                if step == 60 and fv < 0.3:
                    break

            # Phase 2: L-BFGS for fine-tuning (converges faster near optimum)
            if fv > 0.5:
                params_lbfgs = params.detach().clone().requires_grad_(True)

                def closure():
                    opt_l.zero_grad()
                    Uc = build_unitary(params_lbfgs, nq, cx_pairs)
                    f = proc_fidelity(U_target_t, Uc)
                    loss = 1.0 - f
                    loss.backward()
                    return loss

                opt_l = torch.optim.LBFGS([params_lbfgs], lr=1.0,
                                           max_iter=20, history_size=10,
                                           line_search_fn="strong_wolfe")
                for _ in range(5):
                    opt_l.step(closure)
                    Uc = build_unitary(params_lbfgs, nq, cx_pairs)
                    fv = proc_fidelity(U_target_t, Uc).item()
                    if fv >= FIDELITY_THR:
                        return fv, True, params_lbfgs.detach()

                if fv > best_fid:
                    best_fid = fv
                    best_params = params_lbfgs.detach()
            else:
                if fv > best_fid:
                    best_fid = fv
                    best_params = params.detach()

    return best_fid, False, best_params


def compile_unitary(U_np, nq, bl_2q, time_limit=None, t0=None, minimize=False,
                    per_unitary_budget=None, use_rl=False):
    """
    Find CX count that beats Qiskit baseline.
    Phase 0 (if use_rl and 3q): try RL agent first.
    Phase 1: quick beat — try bl_2q-1.
    Phase 2 (if minimize=True): search downward with warm-starting + L-BFGS.
    Returns (n_cx_best, fidelity, success).
    """
    global _RL_POLICY
    t_uni_start = time.time()
    U_t = torch.tensor(U_np, dtype=torch.complex128)
    patterns = get_patterns(nq)

    if bl_2q <= 0:
        return 0, 1.0, False

    # Diagonal unitaries at known optimal CX can't be beaten
    is_diag = np.max(np.abs(U_np - np.diag(np.diag(U_np)))) < 1e-10
    if is_diag and bl_2q <= 2 ** nq - 2:
        return bl_2q, 1.0, False

    target_cx = bl_2q - 1

    # Phase 0: Try RL agent for 3q if available — use RL pattern + variational minimize
    if use_rl and nq == 3 and _RL_POLICY is not None:
        rl_cx, rl_fid, rl_ok = rl_compile(_RL_POLICY, U_np, nq,
                                            max_cx=target_cx, n_rollouts=1)
        if rl_ok and rl_cx < bl_2q:
            if not minimize:
                return rl_cx, rl_fid, True
            # Use RL result as starting point, continue minimize from there
            best_cx = rl_cx
            best_fid = rl_fid
            # Try lower CX with the standard variational approach
            for n_cx in range(rl_cx - 1, -1, -1):
                if per_unitary_budget and (time.time() - t_uni_start) > per_unitary_budget:
                    break
                fid2, ok2, _ = try_compile(U_t, nq, n_cx, patterns[:1],
                                            n_trials=2, n_steps=80, lr=0.1)
                if ok2:
                    best_cx = n_cx
                    best_fid = fid2
                else:
                    if fid2 < 0.5:
                        break
            return best_cx, best_fid, True

    # Adjust budget by qubit count
    if nq <= 3:
        trials1, steps1, lr1 = 3, 120, 0.1
    elif nq == 4:
        trials1, steps1, lr1 = 2, 150, 0.1
    else:
        trials1, steps1, lr1 = 1, 200, 0.1

    # Phase 1: beat the baseline (Adam + L-BFGS)
    fid, ok, best_params = try_compile(U_t, nq, target_cx, patterns,
                                        n_trials=trials1, n_steps=steps1, lr=lr1)
    if not ok:
        return bl_2q, fid, False

    if not minimize:
        return target_cx, fid, True

    # Phase 2: linear minimize with warm-starting (use only 1st pattern for speed)
    best_cx = target_cx
    best_fid = fid
    prev_params = best_params
    min_patterns = patterns[:1]  # single pattern for minimize speed

    for n_cx in range(target_cx - 1, -1, -1):
        if per_unitary_budget and (time.time() - t_uni_start) > per_unitary_budget:
            break
        if time_limit and t0 and time.time() - t0 > time_limit:
            break

        fid2, ok2, params2 = try_compile(
            U_t, nq, n_cx, min_patterns,
            n_trials=2, n_steps=80, lr=0.1,
            init_params=prev_params,
        )
        if ok2:
            best_cx = n_cx
            best_fid = fid2
            prev_params = params2
        else:
            if fid2 < 0.5:
                break

    return best_cx, best_fid, True


# ─── Data Loading ───

def load_benchmarks(path="benchmark_data.pt"):
    if not os.path.exists(path):
        from scipy.stats import unitary_group
        benchmarks = {}
        for i in range(30):
            U = unitary_group.rvs(8)
            benchmarks[f"synth_{i}"] = {
                "unitary": U, "n_qubits": 3, "category": "haar",
                "baselines": {},
            }
        return benchmarks, {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["benchmarks"], data.get("kak_baselines", {})


# ─── Progress tracking (from plot_progress.py) ───
from plot_progress import record_iteration, save_progress


# ─── Parallel compilation worker ───

def _compile_worker(args):
    """Worker function for parallel compilation."""
    if len(args) == 7:
        label, U_np, nq, bl_2q, do_min, per_budget, use_rl = args
    else:
        label, U_np, nq, bl_2q, do_min, per_budget = args
        use_rl = False
    torch.manual_seed(42)
    np.random.seed(42)
    n_cx, fid, ok = compile_unitary(
        U_np, nq, bl_2q, minimize=do_min, per_unitary_budget=per_budget,
        use_rl=use_rl,
    )
    beat = ok and n_cx < bl_2q
    return {
        "label": label, "beat": beat, "success": ok,
        "ours": n_cx, "qiskit": bl_2q, "fid": fid,
    }


# ─── Main ───

def evaluate_qubit_group(benchmarks, nq, basis, t0, results, time_limit=MAX_RUN_TIME):
    """Evaluate all test unitaries for a given qubit count. Mutates results list."""
    keys = [k for k, v in benchmarks.items() if v["n_qubits"] == nq]
    random.shuffle(keys)
    split = max(1, int(0.8 * len(keys)))
    test_keys = keys[split:]
    print(f"\n{nq}-qubit: {len(keys)} total, {len(test_keys)} test")

    n_beat = 0
    n_eval = 0

    for label in test_keys:
        if time.time() - t0 > time_limit:
            print("  [TIME LIMIT]")
            break

        info = benchmarks[label]
        U_target = info["unitary"]
        cat = info.get("category", "?")

        bl = info.get("baselines", {}).get(f"{basis}_opt3", {})
        bl_2q = bl.get("two_qubit_gates", 999) if bl.get("success") else 999

        if bl_2q <= 0:
            results.append({"label": label, "beat": False, "success": False,
                            "ours": 0, "qiskit": bl_2q, "fid": 1.0, "category": cat})
            n_eval += 1
            print(f"  [{len(results):3d}] {label:35s} SKIP  qiskit={bl_2q:3d}")
            continue

        t_uni = time.time()
        # 3q: CEM+minimize with 5s budget; 4q: Adam minimize with 10s budget
        do_min = True
        per_budget = 5.0 if nq <= 3 else 10.0
        n_cx, fid, ok = compile_unitary(
            U_target, nq, bl_2q,
            time_limit=time_limit, t0=t0, minimize=do_min,
            per_unitary_budget=per_budget,
        )
        dt = time.time() - t_uni

        beat = ok and n_cx < bl_2q
        n_eval += 1
        if beat:
            n_beat += 1

        tag = "BEAT" if beat else ("OK" if ok else "FAIL")
        print(f"  [{len(results)+1:3d}] {label:35s} {tag:4s}  ours={n_cx:3d}  qiskit={bl_2q:3d}  "
              f"fid={fid:.4f}  cat={cat}  {dt:.1f}s")

        results.append({
            "label": label, "beat": beat, "success": ok,
            "ours": n_cx, "qiskit": bl_2q, "fid": fid, "category": cat, "nq": nq,
        })

    return n_beat, n_eval


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 70)
    print("QUANTUM CIRCUIT COMPILATION — VARIATIONAL OPTIMIZATION")
    print("=" * 70)

    benchmarks, _ = load_benchmarks()
    print(f"Loaded {len(benchmarks)} benchmarks")

    basis = "ibm"
    results = []
    total_beat = 0
    total_eval = 0

    # Phase 1: Parallel 3-qubit compilation (Adam + L-BFGS + minimize)
    keys_3q = [k for k, v in benchmarks.items() if v["n_qubits"] == 3]
    random.shuffle(keys_3q)
    split_3q = max(1, int(0.8 * len(keys_3q)))
    test_3q = keys_3q[split_3q:]
    print(f"\n3-qubit: {len(keys_3q)} total, {len(test_3q)} test (parallel)")

    work_items = []
    for label in test_3q:
        info = benchmarks[label]
        bl = info.get("baselines", {}).get(f"{basis}_opt3", {})
        bl_2q = bl.get("two_qubit_gates", 999) if bl.get("success") else 999
        work_items.append((label, info["unitary"], 3, bl_2q, True, 6.0))

    t_par = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        par_results = list(pool.map(_compile_worker, work_items))
    dt_par = time.time() - t_par

    for r in par_results:
        r["category"] = benchmarks[r["label"]].get("category", "?")
        r["nq"] = 3
        results.append(r)
        total_eval += 1
        if r["beat"]:
            total_beat += 1
        tag = "BEAT" if r["beat"] else ("OK" if r["success"] else "FAIL")
        print(f"  [{total_eval:3d}] {r['label']:35s} {tag:4s}  ours={r['ours']:3d}  "
              f"qiskit={r['qiskit']:3d}  fid={r['fid']:.4f}  cat={r['category']}")
    print(f"  3q done: {total_beat}/{total_eval} in {dt_par:.1f}s")

    # Phase 2: Parallel 4-qubit compilation
    remaining = MAX_RUN_TIME - (time.time() - t0)
    if remaining > 30:
        print(f"\n  [{remaining:.0f}s remaining, starting 4-qubit (parallel)]")
        keys_4q = [k for k, v in benchmarks.items() if v["n_qubits"] == 4]
        random.shuffle(keys_4q)
        split_4q = max(1, int(0.8 * len(keys_4q)))
        test_4q = keys_4q[split_4q:]
        print(f"  4-qubit: {len(keys_4q)} total, {len(test_4q)} test")

        work_items_4q = []
        for label in test_4q:
            info = benchmarks[label]
            bl = info.get("baselines", {}).get(f"{basis}_opt3", {})
            bl_2q = bl.get("two_qubit_gates", 999) if bl.get("success") else 999
            work_items_4q.append((label, info["unitary"], 4, bl_2q, True, 25.0))

        t_par4 = time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            par_results_4q = list(pool.map(_compile_worker, work_items_4q))
        dt_par4 = time.time() - t_par4

        for r in par_results_4q:
            r["category"] = benchmarks[r["label"]].get("category", "?")
            r["nq"] = 4
            results.append(r)
            total_eval += 1
            if r["beat"]:
                total_beat += 1
            tag = "BEAT" if r["beat"] else ("OK" if r["success"] else "FAIL")
            print(f"  [{total_eval:3d}] {r['label']:35s} {tag:4s}  ours={r['ours']:3d}  "
                  f"qiskit={r['qiskit']:3d}  fid={r['fid']:.4f}  cat={r['category']}")
        print(f"  4q done in {dt_par4:.1f}s")

    test_acc = total_beat / total_eval if total_eval > 0 else 0.0
    elapsed = time.time() - t0

    # Summary stats
    successful = [r for r in results if r.get("success")]
    if successful:
        avg_ours = np.mean([r["ours"] for r in successful])
        avg_qiskit = np.mean([r["qiskit"] for r in successful])
        print(f"\nAvg CX: ours={avg_ours:.1f}  qiskit={avg_qiskit:.1f}  "
              f"saving={avg_qiskit-avg_ours:.1f} ({(avg_qiskit-avg_ours)/avg_qiskit*100:.0f}%)")

    history = record_iteration(results, test_acc, "iter25")
    save_progress(results, test_acc, history=history)
    print(f"\n{'=' * 70}")
    print(f"Beat {total_beat}/{total_eval} unitaries in {elapsed:.1f}s")
    print(f"RESULT: TEST_ACC={test_acc:.4f}")


if __name__ == "__main__":
    main()
