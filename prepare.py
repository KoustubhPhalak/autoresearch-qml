"""
Lattice Reduction Benchmarking for LWE Concrete Security — Primal & Dual
==========================================================================
Runs ONCE before the autonomous experiment loop.

Attacks implemented:
  1. Primal uSVP (Kannan embedding → BKZ)
  2. Dual distinguishing (dual lattice → BKZ → inner-product test)

NOT implemented (noted for honesty):
  - Hybrid meet-in-the-middle
  - Arora–Ge algebraic attack
  These would require substantially more code and are left as future work.
  The autoresearch loop operates only on the primal+dual data.

Secret distribution:
  Secrets are sampled as SMALL vectors (CBD or rounded Gaussian), matching
  the actual Kyber/ML-KEM parameter structure. This is critical because
  the primal embedding target vector is (e, -s, 1), so a uniform-mod-q
  secret would give a completely wrong attack geometry.

Saves everything to  lattice_benchmark.pt  for the cost-model fitting agent.

Hardware note:
  CPU-bound. RTX 4090 GPU is reserved for train.py model fitting.
  Expected runtime: 2–5 hours depending on CPU and fpylll availability.
"""

import os
import time
import math
import json
import warnings
import numpy as np
from scipy.linalg import qr
from pathlib import Path

import torch

# ─────────────────────────────────────────────
# Configuration — DO NOT MODIFY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# LWE parameter grid
# Lattice dimension for primal = m + n + 1, for dual = m
DIMENSIONS = [40, 50, 60]                        # main grid (full BKZ sweep)
EXTRAP_DIMENSIONS = [70, 80]                      # extrapolation targets (lighter sweep)

INSTANCES_PER_CONFIG = 5

# Modulus choices per dimension n
MODULUS_TAGS = ["small", "large"]

# Error distributions
#   ("gaussian", σ)     — discrete Gaussian with parameter σ
#   ("binomial", η)     — centered binomial distribution CBD(η), as in Kyber
ERROR_CONFIGS = [
    ("gaussian", 1.0),     # small error
    ("gaussian", 3.19),    # matches Kyber-512 σ equivalent
    ("binomial", 2),       # CBD(2), used in Kyber-512/768/1024
]

# BKZ block sizes
BKZ_BETAS = [10, 15, 20, 25, 30, 35]
BKZ_MAX_TOURS = 8
BKZ_TIMEOUT = 300           # seconds per individual BKZ run

# Attack types to run
ATTACK_TYPES = ["primal", "dual"]


# ─────────────────────────────────────────────
# Prime generation
# ─────────────────────────────────────────────
def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def next_prime(n):
    n = max(2, int(math.ceil(n)))
    while not is_prime(n):
        n += 1
    return n

def modulus_for_dim(n, tag):
    if tag == "small":
        return next_prime(int(math.ceil(n ** 1.5)))
    elif tag == "large":
        return next_prime(n * n)
    else:
        raise ValueError(f"Unknown modulus tag: {tag}")


# ─────────────────────────────────────────────
# LWE Instance Generation
# ─────────────────────────────────────────────
def sample_discrete_gaussian(size, sigma):
    """Sample from discrete Gaussian over Z with parameter σ."""
    return np.round(np.random.normal(0, sigma, size=size)).astype(np.int64)

def sample_centered_binomial(size, eta):
    """
    CBD(η): x = Σ_{i=1}^{η} (a_i - b_i), a_i,b_i ~ Bernoulli(0.5).
    Used in Kyber/ML-KEM for both secrets and errors.
    """
    a = np.random.randint(0, 2, size=(size, eta))
    b = np.random.randint(0, 2, size=(size, eta))
    return (a.sum(axis=1) - b.sum(axis=1)).astype(np.int64)

def _sample_small_vector(size, error_type, error_param):
    """Sample a small vector (used for both secrets and errors)."""
    if error_type == "gaussian":
        return sample_discrete_gaussian(size, error_param)
    elif error_type == "binomial":
        return sample_centered_binomial(size, int(error_param))
    else:
        raise ValueError(f"Unknown error type: {error_type}")

def generate_lwe_instance(n, m, q, error_type, error_param):
    """
    Generate an LWE instance: (A, b, s, e) where b = A·s + e mod q.

    IMPORTANT: Both s and e are sampled as small vectors (not uniform mod q).
    This matches the Kyber/ML-KEM parameter structure where the secret
    is drawn from the same distribution as the error.
    """
    A = np.random.randint(0, q, size=(m, n), dtype=np.int64)

    # Small secret — same distribution as error (Kyber-style "normal form")
    s = _sample_small_vector(n, error_type, error_param)
    e = _sample_small_vector(m, error_type, error_param)

    b = (A @ s + e) % q

    return {
        "A": A, "b": b, "s": s, "e": e,
        "n": n, "m": m, "q": q,
        "error_type": error_type,
        "error_param": float(error_param),
    }

def optimal_m(n, q):
    """Rough optimal number of samples for primal uSVP."""
    return max(n, min(2 * n, int(math.ceil(n * math.log(q) / math.log(max(q / 2, 2))))))


# ─────────────────────────────────────────────
# Primal uSVP Lattice Construction
# ─────────────────────────────────────────────
def build_primal_lattice(A, b, q, n, m):
    """
    Kannan embedding for the primal uSVP attack.

        B = [ q·I_m   0     0 ]   ← m rows
            [ A^T     I_n   0 ]   ← n rows
            [ b^T     0     1 ]   ← 1 row

    Dimension: d = m + n + 1.
    Target short vector: (e, -s, 1) with norm ≈ sqrt(||e||² + ||s||² + 1).
    Lattice determinant: q^m (the top-left block contributes q^m, rest is unimodular).
    """
    d = m + n + 1
    B = np.zeros((d, d), dtype=np.int64)
    B[:m, :m] = q * np.eye(m, dtype=np.int64)
    B[m:m+n, :m] = A.T
    B[m:m+n, m:m+n] = np.eye(n, dtype=np.int64)
    B[m+n, :m] = b
    B[m+n, m+n] = 1
    return B

def primal_log_volume(q, m):
    """
    Log determinant of the primal lattice.
    det(B) = q^m (from the q·I_m block; remaining blocks are unimodular).
    """
    return m * math.log(q)


# ─────────────────────────────────────────────
# Dual Distinguishing Attack
# ─────────────────────────────────────────────
def _modinv(a, m):
    """Modular inverse of a mod m using extended Euclidean algorithm."""
    if m == 1:
        return 0
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        return None  # no inverse
    return x % m

def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def build_dual_lattice(A, q, n, m):
    """
    Build an m×m basis for Λ^⊥_q(A^T) = {w ∈ Z^m : A^T w ≡ 0 mod q}.

    Method: Gaussian elimination on A^T mod q to find the kernel, then
    combine kernel vectors with q·e_j for pivot columns.

    The dual distinguishing attack uses short vectors w in this lattice:
      w^T b = w^T (As + e) = w^T e  (mod q)
    which is small when w and e are both short, versus uniform for random b.

    The lattice has dimension m and volume q^{rank(A^T mod q)} = q^{min(n,m)}.
    """
    AT = (A.T % q).astype(np.int64)  # n × m matrix

    # Row-reduce A^T mod q
    aug = AT.copy()
    n_rows, n_cols = aug.shape
    pivot_cols = []
    pivot_row_idx = 0

    for col in range(n_cols):
        if pivot_row_idx >= n_rows:
            break
        # Find a row with non-zero entry in this column
        found = -1
        for row in range(pivot_row_idx, n_rows):
            if aug[row, col] % q != 0:
                found = row
                break
        if found == -1:
            continue

        # Swap to pivot position
        if found != pivot_row_idx:
            aug[[pivot_row_idx, found]] = aug[[found, pivot_row_idx]]

        # Scale pivot row so pivot entry = 1 mod q
        piv = int(aug[pivot_row_idx, col]) % q
        inv = _modinv(piv, q)
        if inv is None:
            continue  # skip if not invertible (shouldn't happen for prime q)
        aug[pivot_row_idx] = (aug[pivot_row_idx] * inv) % q

        # Eliminate this column in all other rows
        for row in range(n_rows):
            if row != pivot_row_idx and aug[row, col] % q != 0:
                factor = int(aug[row, col]) % q
                aug[row] = (aug[row] - factor * aug[pivot_row_idx]) % q

        pivot_cols.append(col)
        pivot_row_idx += 1

    # Now aug is in reduced row echelon form mod q
    # pivot_cols: columns where A^T has rank (these get q·e_j basis vectors)
    # free_cols: remaining columns (these get kernel vectors)
    free_cols = [c for c in range(m) if c not in pivot_cols]
    rank_AT = len(pivot_cols)

    # Build m basis vectors for Λ^⊥_q(A^T)
    basis_vectors = []

    # For each free column: construct a kernel vector
    # v[free_col] = 1, v[pivot_col_i] = -aug[i, free_col] mod q
    for fc in free_cols:
        v = np.zeros(m, dtype=np.int64)
        v[fc] = 1
        for i, pc in enumerate(pivot_cols):
            v[pc] = (-int(aug[i, fc])) % q
        basis_vectors.append(v)

    # For each pivot column: q·e_j is in the kernel
    for pc in pivot_cols:
        v = np.zeros(m, dtype=np.int64)
        v[pc] = q
        basis_vectors.append(v)

    B = np.array(basis_vectors, dtype=np.int64)
    assert B.shape == (m, m), f"Dual basis should be {m}×{m}, got {B.shape}"
    return B, rank_AT


def dual_log_volume(q, rank_AT):
    """
    Log volume of Λ^⊥_q(A^T).
    Volume = q^{rank(A^T mod q)}, using the actual rank from row reduction.
    """
    return rank_AT * math.log(q)

def check_dual_distinguishing(B_reduced, b, b_random, q, sigma_eff, max_vecs=10):
    """
    Dual distinguishing: quantitative bias measurement (HEURISTIC).

    For w ∈ Λ^⊥_q(A^T):
      - w^T b_lwe = w^T e mod q  → biased toward 0
      - w^T b_rand mod q         → uniform on Z_q

    Scans up to max_vecs basis vectors, skipping trivial q-multiple vectors
    (where all entries are divisible by q — these give ⟨w,b⟩ ≡ 0 mod q
    for ANY b, not just LWE, so they have zero distinguishing power).

    Uses the shortest non-trivial vector found.

    Returned fields:
      inner_lwe:       centered residue ⟨w, b_lwe⟩ mod q, in [-q/2, q/2]
      inner_random:    centered residue ⟨w, b_rand⟩ mod q
      w_norm:          ||w||
      normalized_score: |inner_lwe| / (σ · ||w||), lower = more distinguishing
      heuristic_success: True if normalized_score < 3.0 (very rough threshold)
      is_trivial:      True if no non-trivial vector was found
    """
    d = B_reduced.shape[0]
    m = min(B_reduced.shape[1] if B_reduced.ndim > 1 else len(B_reduced), len(b))

    # Find the shortest non-trivial basis vector
    best_w = None
    best_norm = float("inf")

    for i in range(min(max_vecs, d)):
        w = B_reduced[i][:m].astype(np.int64) if B_reduced.ndim > 1 else B_reduced[:m].astype(np.int64)

        # Skip trivial q-multiple vectors: all entries divisible by q
        if np.all(w % q == 0):
            continue

        w_norm = float(np.linalg.norm(w.astype(np.float64)))
        if w_norm < best_norm:
            best_w = w
            best_norm = w_norm

        if B_reduced.ndim == 1:
            break  # single vector passed

    if best_w is None:
        # All vectors were trivial q-multiples
        return {
            "inner_lwe": 0, "inner_random": 0,
            "w_norm": 0.0, "normalized_score": float("inf"),
            "heuristic_success": False, "is_trivial": True,
        }

    w = best_w
    w_norm = best_norm

    # LWE inner product
    inner_lwe = int(np.dot(w, b[:m].astype(np.int64))) % q
    if inner_lwe > q // 2:
        inner_lwe -= q

    # Random baseline inner product
    inner_rand = int(np.dot(w, b_random[:m].astype(np.int64))) % q
    if inner_rand > q // 2:
        inner_rand -= q

    # Normalized score: |⟨w, b⟩| / (σ · ||w||)
    denom = max(sigma_eff * w_norm, 1e-10)
    normalized_score = abs(inner_lwe) / denom

    return {
        "inner_lwe": inner_lwe,
        "inner_random": inner_rand,
        "w_norm": w_norm,
        "normalized_score": float(normalized_score),
        "heuristic_success": normalized_score < 3.0,
        "is_trivial": False,
    }


# ─────────────────────────────────────────────
# Attack Success Verification
# ─────────────────────────────────────────────
def check_primal_success(B_reduced, e_true, s_true, q, n, m):
    """
    Exact check: does the shortest vector in the reduced basis
    match ±(e, -s, 1) modulo q?

    No heuristic fallbacks — either it matches or it doesn't.
    """
    d = B_reduced.shape[0]

    for i in range(min(5, d)):
        bi = B_reduced[i]

        # Check if last coordinate is ±1 (embedding coordinate)
        if abs(bi[-1]) != 1:
            continue

        sign = int(bi[-1])
        e_cand = sign * bi[:m]
        s_cand = -sign * bi[m:m+n]

        # Exact mod-q comparison
        e_match = np.all((e_cand - e_true) % q == 0)
        s_match = np.all((s_cand - s_true) % q == 0)

        if e_match and s_match:
            return True

    return False


# ─────────────────────────────────────────────
# Lattice Reduction: fpylll backend
# ─────────────────────────────────────────────
HAVE_FPYLLL = False
_FPYLLL_BKZ_MODE = None

try:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO
    from fpylll import BKZ as BKZ_module
    HAVE_FPYLLL = True

    try:
        from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
        _test_par = BKZ_module.Param(block_size=10, max_loops=1,
                                      strategies=BKZ_module.DEFAULT_STRATEGY)
        _FPYLLL_BKZ_MODE = "bkz2"
        print("[prepare] fpylll: BKZ 2.0 with strategies.")
    except Exception:
        try:
            from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
            _test_par = BKZ_module.Param(block_size=10, max_loops=1)
            _FPYLLL_BKZ_MODE = "bkz2_no_strat"
            print("[prepare] fpylll: BKZ 2.0 (no strategies file).")
        except Exception:
            _FPYLLL_BKZ_MODE = "plain"
            print("[prepare] fpylll: plain BKZ.reduction.")

except ImportError:
    print("[prepare] fpylll not found — using numpy fallback (slower, approximate).")


def _fpylll_reduce(B_np, beta, max_tours=BKZ_MAX_TOURS, timeout=BKZ_TIMEOUT):
    """Run LLL + BKZ-β using fpylll with automatic mode fallback."""
    d, ncols = B_np.shape
    M = IntegerMatrix(d, ncols)
    for i in range(d):
        for j in range(ncols):
            M[i, j] = int(B_np[i, j])

    t0 = time.perf_counter()
    LLL.reduction(M)
    t_lll = time.perf_counter() - t0

    effective_beta = min(beta, d)
    if effective_beta < 2:
        B_red = np.array([[M[i, j] for j in range(ncols)] for i in range(d)], dtype=np.int64)
        b1_norm = np.linalg.norm(B_red[0].astype(np.float64))
        return {
            "basis": B_red, "b1_norm": float(b1_norm),
            "lll_time": t_lll, "bkz_time": 0.0, "total_time": t_lll,
            "svp_calls_est": 0, "tours": 0, "success": True,
        }

    t1 = time.perf_counter()
    success = False

    # Try BKZ2 with strategies
    if _FPYLLL_BKZ_MODE == "bkz2":
        try:
            par = BKZ_module.Param(block_size=effective_beta, max_loops=max_tours,
                                    strategies=BKZ_module.DEFAULT_STRATEGY)
            bkz_obj = BKZ2(GSO.Mat(M))
            bkz_obj(par)
            success = True
        except Exception:
            pass

    # Try BKZ2 without strategies
    if not success and _FPYLLL_BKZ_MODE in ("bkz2", "bkz2_no_strat"):
        try:
            par = BKZ_module.Param(block_size=effective_beta, max_loops=max_tours)
            bkz_obj = BKZ2(GSO.Mat(M))
            bkz_obj(par)
            success = True
        except Exception:
            pass

    # Plain BKZ fallback
    if not success:
        try:
            par = BKZ_module.Param(block_size=effective_beta, max_loops=max_tours)
            BKZ.reduction(M, par)
            success = True
        except Exception as e:
            warnings.warn(f"All fpylll BKZ modes failed (β={beta}): {e}")

    t_bkz = time.perf_counter() - t1
    B_red = np.array([[M[i, j] for j in range(ncols)] for i in range(d)], dtype=np.int64)
    b1_norm = np.linalg.norm(B_red[0].astype(np.float64))

    # NOTE: svp_calls is an ESTIMATE, not a measured count from the solver.
    svp_calls_est = max(0, (d - effective_beta + 1)) * max_tours

    return {
        "basis": B_red, "b1_norm": float(b1_norm),
        "lll_time": t_lll, "bkz_time": t_bkz,
        "total_time": t_lll + t_bkz,
        "svp_calls_est": svp_calls_est,
        "tours": max_tours, "success": success,
    }


# ─────────────────────────────────────────────
# Lattice Reduction: Numpy fallback
# ─────────────────────────────────────────────
def _gram_schmidt(B):
    n = B.shape[0]
    B_star = np.zeros_like(B, dtype=np.float64)
    mu = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        B_star[i] = B[i].astype(np.float64)
        for j in range(i):
            norm_sq = np.dot(B_star[j], B_star[j])
            if norm_sq < 1e-30:
                mu[i, j] = 0.0
            else:
                mu[i, j] = np.dot(B[i].astype(np.float64), B_star[j]) / norm_sq
            B_star[i] -= mu[i, j] * B_star[j]
    return B_star, mu


def _lll_reduce_numpy(B_in, delta=0.99):
    """LLL reduction in pure numpy."""
    B = B_in.copy().astype(np.int64)
    n = B.shape[0]
    B_star, mu = _gram_schmidt(B)
    k = 1
    max_iters = n * n * 20

    for _ in range(max_iters):
        if k >= n:
            break
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                r = int(round(mu[k, j]))
                B[k] -= r * B[j]
                B_star, mu = _gram_schmidt(B)

        b_star_k_sq = np.dot(B_star[k], B_star[k])
        b_star_k1_sq = np.dot(B_star[k-1], B_star[k-1])

        if b_star_k_sq >= (delta - mu[k, k-1]**2) * b_star_k1_sq:
            k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]
            B_star, mu = _gram_schmidt(B)
            k = max(k - 1, 1)

    return B


def _numpy_bkz_reduce(B_in, beta, max_tours=BKZ_MAX_TOURS, timeout=BKZ_TIMEOUT):
    """Simplified BKZ-β using numpy LLL on local blocks."""
    d = B_in.shape[0]
    effective_beta = min(beta, d)

    t0 = time.perf_counter()
    B = _lll_reduce_numpy(B_in)
    t_lll = time.perf_counter() - t0

    if effective_beta <= 2:
        b1_norm = np.linalg.norm(B[0].astype(np.float64))
        return {
            "basis": B, "b1_norm": float(b1_norm),
            "lll_time": t_lll, "bkz_time": 0.0, "total_time": t_lll,
            "svp_calls_est": 0, "tours": 0, "success": True,
        }

    t1 = time.perf_counter()
    svp_calls = 0
    tours_done = 0

    for tour in range(max_tours):
        if time.perf_counter() - t0 > timeout:
            break
        old_norm = np.linalg.norm(B[0].astype(np.float64))

        for kappa in range(d - 1):
            end = min(kappa + effective_beta, d)
            if end - kappa < 2:
                continue
            B_local = B[kappa:end].copy()
            B[kappa:end] = _lll_reduce_numpy(B_local)
            svp_calls += 1
            if time.perf_counter() - t0 > timeout:
                break

        B = _lll_reduce_numpy(B)
        new_norm = np.linalg.norm(B[0].astype(np.float64))
        tours_done = tour + 1
        if new_norm >= old_norm * 0.999 and tour > 1:
            break

    t_bkz = time.perf_counter() - t1
    b1_norm = np.linalg.norm(B[0].astype(np.float64))

    return {
        "basis": B, "b1_norm": float(b1_norm),
        "lll_time": t_lll, "bkz_time": t_bkz,
        "total_time": t_lll + t_bkz,
        "svp_calls_est": svp_calls,
        "tours": tours_done, "success": True,
    }


def reduce_lattice(B_np, beta, max_tours=BKZ_MAX_TOURS, timeout=BKZ_TIMEOUT):
    """Dispatch to fpylll or numpy fallback."""
    if HAVE_FPYLLL:
        return _fpylll_reduce(B_np, beta, max_tours, timeout)
    else:
        return _numpy_bkz_reduce(B_np, beta, max_tours, timeout)


# ─────────────────────────────────────────────
# Lattice Quality Metrics
# ─────────────────────────────────────────────
def root_hermite_factor(b1_norm, log_vol, d):
    """
    Root Hermite factor δ = (||b_1|| / vol^{1/d})^{1/d}.
    """
    if b1_norm <= 0 or d <= 0:
        return -1.0
    log_b1 = math.log(max(b1_norm, 1e-300))
    log_delta = (log_b1 - log_vol / d) / d
    return math.exp(log_delta)


# ─────────────────────────────────────────────
# Theoretical Predictions
# ─────────────────────────────────────────────
def estimate_required_beta(n, q, sigma, d, m, secret_is_small=True):
    """
    Estimate BKZ block size β to solve primal uSVP (simplified GSA proxy).

    This is a rough approximation, not a precise formula. It uses:
      δ ≈ (β / (2πe))^{1/(2(β-1))}
    which is the leading term of the GSA Hermite factor prediction,
    omitting the (πβ)^{1/β} correction and other refinements.

    Target vector norm (with small secret): σ√(m+n).
    Lattice volume: q^m (exact for Kannan embedding).
    """
    if secret_is_small:
        target_norm = sigma * math.sqrt(m + n)
    else:
        target_norm = sigma * math.sqrt(m)

    log_vol = m * math.log(q)  # det = q^m, exact for Kannan embedding
    log_target = math.log(max(target_norm, 1e-10))

    best_beta = d
    for beta in range(5, d + 1):
        # GSA Hermite factor for BKZ-β
        if beta <= 1:
            continue
        try:
            log_delta = math.log(beta / (2 * math.pi * math.e)) / (2 * (beta - 1))
        except (ValueError, ZeroDivisionError):
            continue

        delta = math.exp(log_delta)
        # Predicted shortest vector length
        log_predicted = d * math.log(max(delta, 1.0001)) + log_vol / d
        if log_predicted <= log_target:
            best_beta = beta
            break

    return best_beta

def core_svp_cost_log2(beta, model="classical_sieve"):
    """Core-SVP cost in log2 for a single SVP call in dimension β."""
    if model == "classical_sieve":
        return 0.292 * beta
    elif model == "quantum_sieve":
        return 0.265 * beta
    elif model == "enumeration":
        return 0.187 * beta * max(1, math.log2(max(beta, 2)))
    else:
        raise ValueError(f"Unknown model: {model}")

def theoretical_predictions(n, q, sigma, d, m):
    """Compute theoretical security estimates for an LWE instance."""
    beta_est = estimate_required_beta(n, q, sigma, d, m, secret_is_small=True)
    return {
        "estimated_beta": beta_est,
        "classical_sieve_log2": core_svp_cost_log2(beta_est, "classical_sieve"),
        "quantum_sieve_log2": core_svp_cost_log2(beta_est, "quantum_sieve"),
        "enumeration_log2": core_svp_cost_log2(beta_est, "enumeration"),
    }


# ─────────────────────────────────────────────
# NIST Parameter Sets (reference, for extrapolation)
# ─────────────────────────────────────────────
# NOTE: Kyber is Module-LWE, not plain LWE. Treating it as plain LWE
# with n = k*256 is a simplification that gives a rough lower bound on
# security. A proper Module-LWE analysis would need module-specific attacks.
NIST_PARAMS = {
    "Kyber-512":  {"n": 512,  "k": 2, "q": 3329, "eta1": 3, "eta2": 2,
                   "module_n": 256, "claimed_bits": 128,
                   "note": "Module-LWE; plain-LWE proxy is approximate"},
    "Kyber-768":  {"n": 768,  "k": 3, "q": 3329, "eta1": 2, "eta2": 2,
                   "module_n": 256, "claimed_bits": 192,
                   "note": "Module-LWE; plain-LWE proxy is approximate"},
    "Kyber-1024": {"n": 1024, "k": 4, "q": 3329, "eta1": 2, "eta2": 2,
                   "module_n": 256, "claimed_bits": 256,
                   "note": "Module-LWE; plain-LWE proxy is approximate"},
}


# ─────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────
def run_benchmarks(save_path="lattice_benchmark.pt"):
    all_results = []
    instance_id = 0

    all_dims = [(n, False) for n in DIMENSIONS] + [(n, True) for n in EXTRAP_DIMENSIONS]

    for n, is_extrap in all_dims:
        tag_label = "EXTRAP" if is_extrap else "MAIN"
        n_inst = INSTANCES_PER_CONFIG
        print(f"\n{'=' * 70}")
        print(f"  DIMENSION n = {n}  [{tag_label}]  ({n_inst} instances)")
        print(f"{'=' * 70}")

        for q_tag in MODULUS_TAGS:
            q = modulus_for_dim(n, q_tag)
            m = optimal_m(n, q)

            for err_type, err_param in ERROR_CONFIGS:
                err_label = f"{err_type}_{'%.2f' % err_param}"
                sigma_eff = err_param if err_type == "gaussian" else math.sqrt(err_param / 2)

                for inst in range(n_inst):
                    instance_id += 1
                    label_base = f"lwe_n{n}_q{q}_{err_label}_inst{inst}"

                    # Generate LWE instance (small secret!)
                    lwe = generate_lwe_instance(n, m, q, err_type, err_param)
                    e_norm = float(np.linalg.norm(lwe["e"]))
                    s_norm = float(np.linalg.norm(lwe["s"]))
                    target_norm = float(np.linalg.norm(
                        np.concatenate([lwe["e"], -lwe["s"], [1]]).astype(np.float64)
                    ))

                    print(f"\n  {label_base}  n={n} q={q}({q_tag}) err={err_label} m={m}")
                    print(f"    ||e||={e_norm:.1f}  ||s||={s_norm:.1f}  ||(e,-s,1)||={target_norm:.1f}")

                    # ── Run attacks ──
                    # Pre-generate random b for dual distinguishing baseline
                    b_random = np.random.randint(0, q, size=m, dtype=np.int64)

                    for attack in ATTACK_TYPES:
                        if attack == "primal":
                            B = build_primal_lattice(lwe["A"], lwe["b"], q, n, m)
                            d = m + n + 1
                            log_vol = primal_log_volume(q, m)
                        elif attack == "dual":
                            B, rank_AT = build_dual_lattice(lwe["A"], q, n, m)
                            d = B.shape[0]
                            log_vol = dual_log_volume(q, rank_AT)
                        else:
                            continue

                        # NOTE: theoretical_predictions uses a primal uSVP model.
                        # For dual rows this is a rough proxy, not a proper dual theory.
                        theory = theoretical_predictions(n, q, sigma_eff, d, m)

                        # Determine which betas to run
                        if is_extrap:
                            betas = [b for b in BKZ_BETAS if b <= min(30, d - 2)]
                        else:
                            betas = [b for b in BKZ_BETAS if b <= min(35, d - 2)]

                        label = f"{label_base}_{attack}"

                        instance_result = {
                            "label": label,
                            "instance_id": instance_id,
                            "instance_group": label_base,  # for proper train/test splitting
                            "attack": attack,
                            "n": n, "m": m, "q": q, "q_tag": q_tag, "d": d,
                            "error_type": err_type,
                            "error_param": float(err_param),
                            "sigma_eff": sigma_eff,
                            "e_norm": e_norm,
                            "s_norm": s_norm,
                            "target_norm": target_norm,
                            "log_vol": float(log_vol),
                            "is_extrap": is_extrap,
                            "theory": theory,
                            "reductions": [],
                        }

                        for beta in betas:
                            print(f"    {attack:6s} β={beta:3d} ... ", end="", flush=True)

                            t_start = time.perf_counter()
                            try:
                                result = reduce_lattice(B.copy(), beta,
                                                        max_tours=BKZ_MAX_TOURS,
                                                        timeout=BKZ_TIMEOUT)
                                wall_time = time.perf_counter() - t_start

                                delta = root_hermite_factor(result["b1_norm"], log_vol, d)

                                # Attack success check
                                if attack == "primal":
                                    solved = check_primal_success(
                                        result["basis"], lwe["e"], lwe["s"], q, n, m
                                    )
                                    dual_stats = None
                                elif attack == "dual":
                                    dual_stats = check_dual_distinguishing(
                                        result["basis"], lwe["b"], b_random,
                                        q, sigma_eff
                                    )
                                    solved = dual_stats["heuristic_success"]
                                else:
                                    solved = False
                                    dual_stats = None

                                red_info = {
                                    "beta": beta,
                                    "wall_time": wall_time,
                                    "b1_norm": result["b1_norm"],
                                    "root_hermite_factor": delta,
                                    "lll_time": result["lll_time"],
                                    "bkz_time": result["bkz_time"],
                                    "svp_calls_est": result["svp_calls_est"],
                                    "tours": result["tours"],
                                    "attack_success": solved,
                                    "attack_success_is_heuristic": (attack == "dual"),
                                    "success": result["success"],
                                }
                                if dual_stats is not None:
                                    red_info["dual_normalized_score"] = dual_stats["normalized_score"]
                                    red_info["dual_inner_lwe"] = dual_stats["inner_lwe"]
                                    red_info["dual_inner_random"] = dual_stats["inner_random"]
                                    red_info["dual_w_norm"] = dual_stats["w_norm"]
                                    red_info["dual_is_trivial"] = dual_stats.get("is_trivial", False)

                                if attack == "primal":
                                    status = "SOLVED" if solved else ("ok" if result["success"] else "FAIL")
                                else:
                                    if dual_stats and dual_stats.get("is_trivial"):
                                        status = "TRIVIAL(q-multiple only)"
                                    elif dual_stats:
                                        score_str = f"score={dual_stats['normalized_score']:.2f}"
                                        status = f"DIST({score_str})" if solved else f"no({score_str})"
                                    else:
                                        status = "?"
                                print(f"δ={delta:.5f}  ||b1||={result['b1_norm']:.1f}  "
                                      f"t={wall_time:.2f}s  {status}")

                            except Exception as e:
                                wall_time = time.perf_counter() - t_start
                                red_info = {
                                    "beta": beta, "wall_time": wall_time,
                                    "b1_norm": -1, "root_hermite_factor": -1,
                                    "lll_time": -1, "bkz_time": -1,
                                    "svp_calls_est": 0, "tours": 0,
                                    "attack_success": False, "success": False,
                                    "error": str(e),
                                }
                                print(f"FAILED ({e})")

                            instance_result["reductions"].append(red_info)

                        all_results.append(instance_result)

    # ── Flatten into tabular form ──
    rows = []
    for inst in all_results:
        for red in inst["reductions"]:
            if not red.get("success", False):
                continue
            rows.append({
                "instance_group": inst["instance_group"],
                "attack": inst["attack"],
                "n": inst["n"], "m": inst["m"], "q": inst["q"],
                "q_tag": inst["q_tag"], "d": inst["d"],
                "error_type": inst["error_type"],
                "error_param": inst["error_param"],
                "sigma_eff": inst["sigma_eff"],
                "log_vol": inst["log_vol"],
                "is_extrap": inst["is_extrap"],
                "beta": red["beta"],
                "wall_time": red["wall_time"],
                "bkz_time": red["bkz_time"],
                "lll_time": red["lll_time"],
                "b1_norm": red["b1_norm"],
                "root_hermite_factor": red["root_hermite_factor"],
                "svp_calls_est": red["svp_calls_est"],
                "attack_success": red["attack_success"],
                "attack_success_is_heuristic": red.get("attack_success_is_heuristic", False),
                "dual_normalized_score": red.get("dual_normalized_score", None),
                "theory_beta_est": inst["theory"]["estimated_beta"],
                "theory_classical_log2": inst["theory"]["classical_sieve_log2"],
                "theory_quantum_log2": inst["theory"]["quantum_sieve_log2"],
            })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total instance×attack combos:  {len(all_results)}")
    print(f"  Total reduction runs (valid):  {len(rows)}")
    print(f"  Main grid dims:                {DIMENSIONS}")
    print(f"  Extrapolation dims:            {EXTRAP_DIMENSIONS}")
    print(f"  Attacks:                       {ATTACK_TYPES}")
    print(f"  Backend:                       {'fpylll' if HAVE_FPYLLL else 'numpy fallback'}")

    for atk in ATTACK_TYPES:
        atk_rows = [r for r in rows if r["attack"] == atk]
        if atk_rows:
            solved = sum(1 for r in atk_rows if r["attack_success"])
            times = [r["wall_time"] for r in atk_rows]
            print(f"\n  [{atk}] {len(atk_rows)} runs, {solved} solved, "
                  f"time range [{min(times):.2f}, {max(times):.2f}]s")

    # ── Save ──
    payload = {
        "instances": all_results,
        "flat_rows": rows,
        "nist_params": NIST_PARAMS,
        "config": {
            "seed": SEED,
            "dimensions": DIMENSIONS,
            "extrap_dimensions": EXTRAP_DIMENSIONS,
            "modulus_tags": MODULUS_TAGS,
            "error_configs": ERROR_CONFIGS,
            "bkz_betas": BKZ_BETAS,
            "bkz_max_tours": BKZ_MAX_TOURS,
            "bkz_timeout": BKZ_TIMEOUT,
            "attack_types": ATTACK_TYPES,
            "backend": "fpylll" if HAVE_FPYLLL else "numpy",
            "notes": [
                "Secrets are small (CBD/Gaussian), NOT uniform mod q.",
                "Primal attack_success is an exact mod-q check.",
                "Dual attack_success is HEURISTIC (normalized_score < 3.0 threshold).",
                "Dual rows include quantitative stats: normalized_score, inner products.",
                "svp_calls_est is a rough estimate, not a solver-side count.",
                "Primal volume = q^m (exact). Dual volume = q^rank(A^T mod q) (exact).",
                "Theory predictions use a simplified GSA proxy for primal uSVP.",
                "Theory column for dual rows is a primal proxy, NOT a proper dual model.",
                "NIST extrapolation treats Module-LWE as plain LWE (approximate).",
                "Hybrid MITM and Arora-Ge attacks are NOT implemented.",
            ],
        },
    }

    torch.save(payload, save_path)
    print(f"\n  Saved to {save_path}")
    print(f"  {len(rows)} data points for cost-model fitting")
    print(f"{'=' * 70}")

    return payload


if __name__ == "__main__":
    run_benchmarks()