"""Regenerate progress.png with iter, major change, and value beside every point."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Full experiment log (iter, acc, status, label)
# label = "iterN: <major change>" for all points
experiments = [
    (0,  0.550, "KEEP",    "iter0: baseline"),
    (1,  0.475, "DISCARD", "iter1: QCNN+CRZ pooling"),
    (2,  0.920, "KEEP",    "iter2: ZZ correlators\n+ trainable basis rotations"),
    (3,  0.920, "DISCARD", "iter3: ZZ+more layers"),
    (4,  0.940, "DISCARD", "iter4: RX-RY-RZ embed"),
    (5,  0.960, "KEEP",    "iter5: StatePrep+probs\n+ MLP(BN+Dropout)"),
    (6,  0.965, "DISCARD", "iter6: multi-basis probs\n(no q-grad)"),
    (7,  0.445, "DISCARD", "iter7: SEL large-init\n(barren plateau)"),
    (8,  0.455, "DISCARD", "iter8: 0.01×randn, no BN"),
    (9,  0.970, "KEEP",    "iter9: zero-init variational\n+ BN(64) head"),
    (10, 0.420, "DISCARD", "iter10: zero-init, no BN"),
    (11, 0.395, "DISCARD", "iter11: 4 Z-expvals\n3-layer circuit"),
    (12, 0.490, "DISCARD", "iter12: CRY zero-init\nshallow head"),
    (13, 0.430, "DISCARD", "iter13: ring CNOT\n+ ZZ correlators"),
    (14, 0.495, "DISCARD", "iter14: ring CNOT\n+ probs, no BN"),
    (15, 0.490, "DISCARD", "iter15: all-CRY, no BN"),
    (16, 0.820, "DISCARD", "iter16: 0.1×randn\n+ BN(64) head"),
    (17, 0.900, "DISCARD", "iter17: all-CRY zero-init\n+ BN head"),
    (18, 0.950, "DISCARD", "iter18: QBN on probs\n(no head BN)"),
    (19, 0.965, "DISCARD", "iter19: dual BN\n(QBN + head BN)"),
    (20, 0.935, "DISCARD", "iter20: true QBN\n(W_bn correction)"),
    (21, 0.925, "DISCARD", "iter21: QBN replaces\nclassical BN"),
    (22, 0.500, "DISCARD", "iter22: 5-qubit ancilla\n(failed)"),
    (23, 0.665, "DISCARD", "iter23: 5-qubit CRY\n+ BN(32) quantum"),
    (24, 0.575, "DISCARD", "iter24: 6-qubit CRY\n+ BN(64) quantum"),
    (25, 0.600, "DISCARD", "iter25: 6-qubit CNOT\n+ BN(64) quantum"),
    (26, 0.545, "DISCARD", "iter26: re-upload params\n(unclassifiable)"),
    (27, 0.895, "DISCARD", "iter27: 3-layer circuit\nbatch=200"),
    (28, 0.935, "DISCARD", "iter28: all-to-all CNOT\ntopology"),
    (29, 0.955, "DISCARD", "iter29: ensemble 2 models\n(avg logits)"),
    (30, 0.975, "KEEP",    "iter30: 120 epochs, lr=0.004\n+ more convergence"),
    (31, 0.420, "DISCARD", "iter31: QBN+Linear(16,2)\n(no capacity)"),
    (32, 0.910, "DISCARD", "iter32: QBN+Linear(16,32)\n+ReLU+Linear(32,2)"),
    (33, 0.935, "DISCARD", "iter33: QBN+Linear(16,64)\n+ReLU+Linear(64,2)"),
    (34, 0.925, "DISCARD", "iter34: QBN+2 hidden layers\n64→32→2 (overfit)"),
    (35, 0.655, "DISCARD", "iter35: QBN+Linear(16,8)\n+ReLU+Linear(8,2)"),
    (36, 0.910, "DISCARD", "iter36: lr=0.01, batch=400\nQBN+16→8→2"),
    (37, 0.965, "DISCARD", "iter37: CBN(not QBN)+Linear(16,256)\n+ReLU+Linear(256,2)"),
    (38, 0.955, "KEEP",    "iter38: True QBN (RY in circuit)\n+Linear(16,256)+ReLU+Linear(256,2)"),
    (39, 0.960, "DISCARD", "iter39: QBN after variational\nlayers (bug fix)"),
    (40, 0.980, "KEEP",    "iter40: corrected CE dataset\n(full power-set subsystems)"),
]

iters  = [e[0] for e in experiments]
accs   = [e[1] for e in experiments]
status = [e[2] for e in experiments]
labels = [e[3] for e in experiments]

fig, ax = plt.subplots(figsize=(22, 10))

# ── Discarded dots ──────────────────────────────────────────────────────────
disc_x = [e[0] for e in experiments if e[2] == "DISCARD"]
disc_y = [e[1] for e in experiments if e[2] == "DISCARD"]
ax.scatter(disc_x, disc_y, c="#aaaaaa", s=60, alpha=0.7, zorder=2, label="Discarded")

# ── Running best step-line ──────────────────────────────────────────────────
kept_x = [e[0] for e in experiments if e[2] == "KEEP"]
kept_y = [e[1] for e in experiments if e[2] == "KEEP"]

# Extend the step-line to the right edge
step_x = kept_x + [max(iters)]
running_best = kept_y[:]
step_y = running_best + [running_best[-1]]

ax.step(step_x, step_y, where="post", color="#2ecc71",
        linewidth=2.5, alpha=0.9, zorder=3, label="Running best")

# ── Kept dots ──────────────────────────────────────────────────────────────
ax.scatter(kept_x, kept_y, c="#2ecc71", s=120, zorder=5,
           edgecolors="#1a7a3a", linewidths=1.5, label="Kept (committed)")

# ── Labels for KEPT points (description + value, rotated above) ────────────
for e in experiments:
    idx, acc, st, lbl = e
    if st == "KEEP" and lbl is not None and idx > 0:   # skip baseline label
        text = f"{lbl}\n{acc:.4f}"
        ax.annotate(text, (idx, acc),
                    textcoords="offset points",
                    xytext=(6, 8), fontsize=8.5,
                    color="#1a7a3a", fontweight="bold",
                    rotation=30, ha="left", va="bottom")

# ── Labels for ALL DISCARDED points ────────────────────────────────────────
# High-acc discarded (>0.85): label goes straight down (away from kept diag labels).
# Low-acc discarded: label goes below-right.
for e in experiments:
    idx, acc, st, lbl = e
    if st == "DISCARD":
        text = f"{lbl}\n{acc:.3f}"
        if acc > 0.85:
            ax.annotate(text, (idx, acc),
                        textcoords="offset points",
                        xytext=(5, -26), fontsize=7.5,
                        color="#555555",
                        ha="left", va="top")
        else:
            ax.annotate(text, (idx, acc),
                        textcoords="offset points",
                        xytext=(5, -22), fontsize=7.5,
                        color="#555555",
                        ha="left", va="top")

# ── Best result annotation ──────────────────────────────────────────────────
best_acc = max(kept_y)
best_iter = kept_x[kept_y.index(best_acc)]
ax.annotate(f"Best: {best_acc:.4f}\niter{best_iter} (true hybrid)",
            xy=(best_iter, best_acc),
            xytext=(best_iter + 1.5, best_acc - 0.03),
            fontsize=9, color="#1a7a3a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a7a3a", lw=1.2))

# ── Reference lines ────────────────────────────────────────────────────────
target = 0.85
baseline = 0.55
ax.axhline(target,   color="#e74c3c", linestyle="--", linewidth=1.2,
           alpha=0.7, label=f"Target ({target})")
ax.axhline(baseline, color="#f39c12", linestyle=":",  linewidth=1.2,
           alpha=0.7, label=f"Old baseline ({baseline})")

# ── Axis labels / title ────────────────────────────────────────────────────
n_total = len(experiments)
n_kept  = sum(1 for e in experiments if e[2] == "KEEP") - 1  # exclude baseline
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("TEST_ACC (higher is better)", fontsize=12)
ax.set_title(
    f"Autoresearch Progress: {n_total} Experiments, {n_kept} Committed Improvements\n"
    "(NTangled: Depth-1 vs Depth-4 SEA State Classification)",
    fontsize=13
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.5, max(iters) + 0.5)
ax.set_ylim(0.28, 1.06)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
