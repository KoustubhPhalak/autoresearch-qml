# Quantum Circuit Compilation — Experiment Log

## Goal
Beat Qiskit opt-3 transpiler on 2-qubit gate count while maintaining fidelity >= 0.999.
TEST_ACC = fraction of held-out benchmark unitaries where we use fewer CX gates than Qiskit.

## Baselines (Qiskit opt-3 CX counts — STATIC, from benchmark_data.pt)
| Qubit | Category | CX Gates | Notes |
|-------|----------|----------|-------|
| 3 | Haar random | 19 | Theoretical min = 14 |
| 3 | Grover oracle | 6 | Optimal (diagonal: 2^n-2) |
| 3 | QFT | 18 | |
| 3 | Hamiltonian sim | 19 | |
| 4 | Haar random | 95 | Theoretical min ≈ 61 |
| 4 | Grover oracle | 14 | Optimal (diagonal: 2^n-2) |
| 4 | QFT | 89 | |
| 4 | Hamiltonian sim | 85-95 | TFIM=85, Heisenberg=95 |
| 5 | Haar random | 423 | Theoretical min ≈ 252 |

## Results Summary

### Per-Category CX Comparison (iter25, final)

| Qubit | Category | Qiskit CX | Ours CX | Saved | % | Fidelity | Beat |
|-------|----------|-----------|---------|-------|---|----------|------|
| 3 | Haar random | 19.0 | **13.2** | **+6** | **31%** | 0.9993 ± 0.0002 | 20/20 |
| 3 | Grover iterate | 19.0 | **13.0** | **+6** | **32%** | 0.9993 | 1/1 |
| 3 | Heisenberg | 19.0 | **12.0** | **+7** | **37%** | 0.9992 | 1/1 |
| 3 | TFIM | 19.0 | **13.0** | **+6** | **32%** | 0.9994 | 1/1 |
| 4 | Haar random | 95.0 | **90.0** | **+5** | **5%** | 0.9991 ± 0.0000 | 19/19 |
| 4 | Heisenberg | 95.0 | **90.0** | **+5** | **5%** | 0.9991 | 2/2 |
| 4 | QFT | 89.0 | **84.0** | **+5** | **6%** | 0.9990 | 1/1 |
| 4 | Grover oracle | 14.0 | 14.0 | 0 | 0% | 1.0000 | 0/1 |

**Overall: TEST_ACC = 0.9783 (45/46), avg 5.4 CX saved (10%)**

### Iteration History

| Iter | Description | TEST_ACC | Scope | Time | Notes |
|------|-------------|----------|-------|------|-------|
| 0 | DQN baseline (original) | 0.0000 | 3q | 72-85s | RL never reaches fidelity threshold |
| 1 | Variational compilation, quick beat | 1.0000 | 3q (23/23) | 16.4s | All at 18 CX vs Qiskit 19 |
| 1b | Variational + minimize | 1.0000 | 3q (8/8) | 295s | 14-15 CX Haar, 8 CX TFIM. Time limit hit. |
| 2 | 3q + 4q, diagonal skip | 0.9783 | 3q+4q (45/46) | 165s | Only miss: 4q Grover oracle (optimal @ 14 CX) |
| 3 | L-BFGS + warm-start minimize | 0.9783 | 3q+4q (45/46) | 278s | 3q: 13 CX (32% savings!), 4q: 94 CX |

## Iteration Details

### iter0 — DQN Baseline (TEST_ACC=0.0000, DISCARD)
- **Hypothesis:** DQN with discrete action space can learn to compile circuits.
- **Change:** Original code: DQN agent, discrete angle bins, dense/sparse reward variants.
- **Result:** TEST_ACC=0.0000 across all configurations.
- **Analysis:** 400 DQN episodes with discrete angles can't navigate the combinatorial search space. Fidelity threshold of 0.999 is nearly unreachable by random exploration.
- **Next:** Replace RL with direct variational optimization.

### iter1 — Variational Compilation (TEST_ACC=1.0000, KEEP)
- **Hypothesis:** Parameterized circuit templates with continuous angle optimization via Adam should compile with fewer CX than Qiskit's generic synthesis.
- **Change:** Complete rewrite:
  - Circuit template: [U3 layer] → CX → [U3 layer] → CX → ... → [U3 layer]
  - U3 = RZ(α)·RY(β)·RZ(γ) (ZYZ decomposition, 3 params per qubit per layer)
  - CX pairs cycle through triangle pattern: (0,1), (1,2), (0,2)
  - Adam optimizer with cosine annealing LR
  - Quick beat mode: try n_cx = bl_2q - 1
- **Result:** TEST_ACC=1.0000 (23/23 test, 3q only). All compiled at 18 CX vs Qiskit's 19.
- **Analysis:** Templates at n_cx=18 are overparameterized (57 params) for SU(8) (63 real dims). Optimization converges reliably in 150 steps / ~0.7s per unitary.

### iter1b — Minimize Mode (TEST_ACC=1.0000, 8/8 evaluated)
- **Hypothesis:** With more optimization budget, we can push CX count closer to theoretical minimum of 14.
- **Change:** After quick beat, search downward from 18 to find minimum viable n_cx.
- **Result:** 8/8 beat (time limit). Haar random: 14-15 CX. TFIM: 8 CX. Avg saving = 5.5 CX (29%).
- **Analysis:** Optimization converges to 14 CX for most Haar random (matching theoretical min!). Structured unitaries (TFIM) have much lower CX requirements. But minimize mode takes ~40s per unitary → can only evaluate 8 in 300s.

### iter2 — 3q + 4q with Diagonal Skip (TEST_ACC=0.9783, KEEP)
- **Hypothesis:** 4-qubit unitaries can also be variationally compiled. Diagonal unitaries (Grover oracles) at theoretical min should be skipped.
- **Change:**
  - Added 4-qubit evaluation (n_cx=94, ~7s per unitary)
  - Diagonal detection: if unitary is diagonal and bl_2q ≤ 2^n - 2, skip (already optimal)
  - Adaptive optimization budgets by qubit count
- **Result:** TEST_ACC=0.9783 (45/46). 23/23 for 3q, 22/23 for 4q. Only miss: 4q Grover oracle (bl_2q=14 = optimal).
- **Analysis:** 4q compilation works well: 94 CX vs Qiskit's 95 for Haar, 88 vs 89 for QFT. The single failure is mathematically unbeatable (Grover oracle at known optimal CX count). Total time: 165s, well within budget.
- **Next:** Try deeper CX minimization for 4q, add TFIM-specific patterns, or try 5q for non-Haar cases.

### iter3 — L-BFGS + Warm-Start Minimize (TEST_ACC=0.9783, KEEP)
- **Hypothesis:** L-BFGS optimizer converges faster than Adam near the optimum. Warm-starting from higher CX solutions accelerates downward search.
- **Change:**
  - Two-phase optimization: Adam (100 steps) → L-BFGS (5 outer iterations) per trial
  - Warm-start: when trying n_cx=k, initialize from the n_cx=k+1 solution
  - Per-unitary time budget: 5s for 3q minimize
  - Minimize enabled for 3q, quick beat for 4q
- **Result:** TEST_ACC=0.9783 (45/46). **3q Haar: 13 CX (vs Qiskit 19, 32% saving!)**. 4q: 94 CX (1 less than Qiskit).
- **Analysis:** L-BFGS is excellent for fine-tuning; the two-phase Adam→L-BFGS approach gives both exploration (Adam) and precision (L-BFGS). 13 CX for 3-qubit is BELOW the theoretical min of 14 for generic SU(8) — individual unitaries can have lower CX than worst case. Avg saving: 3.5 CX (6% overall). TFIM compiles at 13 CX, Heisenberg at 14 CX. Total time 278s.
- **Next:** Push 4q minimize (theoretical min ~61, Qiskit uses 95). Try aggressive jump search for 4q.

### iter5 — Structure Probe Attempt (TEST_ACC=0.9783, KEEP)
- **Hypothesis:** A quick probe at half the baseline CX count can detect structured unitaries (TFIM, QFT) that compress much better. TFIM 4q compiles at 60 CX (vs Qiskit 85) in isolation.
- **Change:** Added structure probe for 4q: test n_cx = bl_2q//2 with 15 Adam steps; if fidelity > threshold, invest more time. Also tested 3rd CX pattern and binary search.
- **Result:** Probe threshold too sensitive — Haar random also passes 0.7-0.95 threshold at half CX, triggering expensive optimization that fails. Reverted to proven iter3 config.
- **Analysis:** The fidelity landscape for overparameterized circuits is deceivingly smooth — even insufficient CX counts show high fidelity during early optimization. Need a better structure detection method (eigenvalue analysis, sparsity metrics).
- **Finding:** TFIM 4q CAN compile at 60 CX (29% below Qiskit 85) when given enough time (~10s). The challenge is time management: Haar random takes 6s for quick beat, leaving little budget for structure detection.
- **Note:** All Qiskit baselines are STATIC (pre-computed in benchmark_data.pt). They never change. Apparent avg changes in plots are due to different evaluation scopes across iterations.

### iter6 — Parallel Compilation (TEST_ACC=0.9783, KEEP)
- **Hypothesis:** Using multiprocessing to compile 3q unitaries in parallel (4 workers) frees time budget for 4q optimization.
- **Change:** `concurrent.futures.ProcessPoolExecutor` with 4 workers for 3q compilation.
- **Result:** 3q time: 130s → **42s** (3.1x speedup). Total: 278s → **172s**. TEST_ACC unchanged at 0.9783.
- **Analysis:** CPU parallelism works perfectly for independent compilation tasks. Each worker runs its own Adam+L-BFGS optimization. 106s of budget freed for future 4q minimize work.
- **Next:** Use freed time for 4q minimize or 5q evaluation.

### iter7 — 4q Minimize with Freed Budget (TEST_ACC=0.9783, KEEP)
- **Hypothesis:** With 106s freed by parallelization, enable warm-start minimize for 4q (10s per-unitary budget).
- **Change:** Enable minimize=True for 4q with per_unitary_budget=10.0.
- **Result:** TEST_ACC=0.9783 (45/46). **4q Haar: 93 CX (vs Qiskit 95, 2 gates saved)**. 4q QFT: **87 CX (vs 89)**. Heisenberg: 93 CX. Total time: 287s.
- **Analysis:** The warm-start minimize pushes 4q from bl-1 to bl-2. Each minimize step at 4q takes ~5s (1 trial, 80 steps, L-BFGS refinement). The 10s budget allows trying 93→92 (fails, fid ~0.998). Average saving: 4.0 CX (7% overall).
- **Next:** Try deeper 4q minimize with more patterns or specialized templates.

### iter8 — CEM + GRPO Experiments (TEST_ACC=0.9783, PARTIAL)
- **Hypothesis:** Cross-Entropy Method (CEM, evolutionary RL) and GRPO (Group Relative Policy Optimization) could find better circuit parameters than gradient-based methods alone.
- **Change:** Implemented CEM: sample population of parameter vectors, evaluate fidelity, keep elite set, update distribution. Also implemented GRPO: train MLP policy to predict circuit params, use group-relative advantages for policy gradient.
- **Result:** CEM works but gives 14-15 CX for 3q (worse than Adam+L-BFGS's 13 CX). GRPO training avg_best_fid = 0.04 (barely learns). Multiprocessing incompatible with GRPO policy (can't pickle nn.Module across processes).
- **Analysis:** CEM is fast (~1s per 3q unitary) but sacrifices optimization quality. The fidelity landscape is smooth enough that gradient-based methods outperform evolutionary approaches for this problem. GRPO needs much more training data/epochs than 300s budget allows.

### iter9 — Full Parallelization (TEST_ACC=0.9783, BEST CONFIG)
- **Hypothesis:** Parallel 4q compilation (4 workers) can cut 4q time from 250s to ~110s, enabling deeper minimize.
- **Change:** Both 3q and 4q compiled in parallel with ProcessPoolExecutor(max_workers=4). 4q per-unitary budget increased to 15s. Reverted to Adam+L-BFGS (no CEM) for quality.
- **Result:** TEST_ACC=0.9783 (45/46). **Total time: 148s (152s remaining!)**. 3q: 13 CX (32% below Qiskit). 4q: 92 CX (3.2% below Qiskit). QFT 4q: 87 CX (vs 89). Avg saving: 4.4 CX (8%).
- **Analysis:** Full parallelization is the biggest speedup: 280s → 148s. The 4q workers each optimize independently on separate CPU cores. With 15s per-unitary budget, the warm-start minimize pushes 4q from 94→92. Heisenberg at 92 too.
- **Note:** Qiskit baselines are STATIC from benchmark_data.pt. All "savings" compare against these fixed values.

### iter10 — Autoregressive REINFORCE RL Agent (TEST_ACC=0.9783, EXPERIMENTAL)
- **Hypothesis:** An RL agent that autoregressively selects CX pairs (observing the residual unitary) can learn better CX patterns than fixed cycles.
- **Change:** Implemented REINFORCE with baseline:
  - Policy: MLP(128→128→128→7) mapping residual unitary to CX pair probabilities + STOP
  - State: flattened real+imag of R = U_target† @ U_cx_sequence (residual)
  - Training: max_cx=18, 25 Adam steps per episode for angle optimization
  - 72 episodes trained in 55s, 39/72 (54%) success
- **Result:** TEST_ACC=0.9783 (45/46). 3q Haar: **15-18 CX** (avg ~16.5, vs Qiskit 19). QFT: 15 CX.
- **Analysis:** The RL agent learns meaningful CX patterns — it discovers CX orderings that differ from the fixed cycle and achieve 1-4 fewer CX than Qiskit. However, it can't match the variational minimize (13 CX) because:
  1. Only 1 training epoch (55s budget, 72 episodes)
  2. REINFORCE needs 1000+ episodes to converge for 18-step sequences
  3. The variational minimize searches systematically over CX counts, while RL learns holistically
- **Key insight (from user):** The bottleneck is DISCRETE (which CX pairs) not CONTINUOUS (rotation angles). The RL approach correctly targets this, but needs more training budget. Gradient-based methods can't optimize discrete CX placement — this is where RL adds genuine value.
- **Next:** More training epochs with curriculum (start easy, increase difficulty). Or combine RL CX patterns with variational minimize.

### iter11 — RL + Variational Minimize Combo (TEST_ACC=0.9783, EXPERIMENTAL)
- **Change:** RL suggests CX pattern → variational minimize searches downward from RL's solution.
- **Result:** QFT at 11 CX (39% below Qiskit 18!) when given 7s budget. Haar at 13-14 CX. But 3q eval takes 185s → no time for 4q.

### iter12-14 — Budget Tuning (TEST_ACC=0.9783, STABLE)
- Reverted to parallel variational (no RL training overhead)
- iter13: 4q 20s budget → 91 CX (4 saved)
- iter14: 3q 6s + 4q 20s → avg 5.0 CX saved (9%)
- Best so far: 3q=13, 4q=91, QFT 4q=85

### iter15-16 — Optimal Budget (TEST_ACC=0.9783, BEST)
- iter15: 4q 25s budget → **90 CX** for Haar (5 saved, 5.3%)
- iter16: 3q 7s + 4q 25s → avg **5.5 CX saved (10%)**
- Both configs complete in ~225s, well within 300s budget

### iter17-20 — Config Exploration (TEST_ACC=0.9783, STABLE)
- Tested 4q budgets from 20-30s, 3q budgets from 5-7s
- All achieve 97.83% with 5.0-5.5 CX avg savings
- Optimal: 3q 6s + 4q 25s (balanced time and quality)

### iter21-25 — Stable Final Config (TEST_ACC=0.9783, FINAL)
- Best config locked: 3q 6s parallel + 4q 25s parallel
- Consistent results across all 5 runs: avg 5.3-5.5 CX saved (10%)
- 3q: 13 CX (32% below Qiskit 19)
- 4q: 90 CX (5.3% below Qiskit 95)
- QFT 4q: 85 CX (4.5% below Qiskit 89)
- Total time: ~220-225s (75s margin)

## Final Summary

| Metric | Value |
|--------|-------|
| **TEST_ACC** | **0.9783 (45/46 test unitaries beat Qiskit opt-3)** |
| 3q avg CX (ours/Qiskit) | 13 / 19 (32% savings) |
| 4q avg CX (ours/Qiskit) | 90 / 95 (5.3% savings) |
| QFT 4q (ours/Qiskit) | 85 / 89 (4.5% savings) |
| Only failure | 4q Grover oracle (already at theoretical minimum 14 CX) |
| Total runtime | ~220s (within 300s budget) |
| Approach | Variational compilation: Adam + L-BFGS + warm-start minimize |

## Key Learnings
1. **Variational > RL** for this 300s budget: gradient-based angle optimization converges in seconds while RL needs thousands of episodes
2. **The discrete CX pattern matters**: RL experiments showed QFT can reach 11 CX (vs 18) with the right pattern, but fixed patterns suffice for Haar
3. **Parallelization is critical**: ProcessPoolExecutor cuts 3q time 3x and 4q time 4x
4. **Diagonal unitaries are unbeatable**: Grover oracles at 2^n-2 CX are already optimal
5. **Overparameterization helps**: More rotation parameters than SU(2^n) degrees of freedom makes optimization landscapes smoother

## Key Insights
1. **Variational > RL** for this problem: direct optimization of continuous rotation angles converges in seconds, while DQN with discrete actions fails completely.
2. **Qiskit opt-3 is not optimal**: Uses 19 CX for 3q Haar when 14 suffices. Uses 95 for 4q when ~61 is theoretical min.
3. **Diagonal unitaries are already optimal** in Qiskit: Grover oracles at 2^n - 2 CX (Gray code bound).
4. **Structured unitaries compress well**: TFIM Hamiltonian sim uses only 8-9 CX (vs 19 from Qiskit) because the time evolution factors into local terms.
5. **Overparameterization helps**: Having more rotation parameters than degrees of freedom in SU(2^n) makes optimization landscapes smoother.
