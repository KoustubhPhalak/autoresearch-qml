"""Generate progress.png for the 8-class NTangled classification experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# All iterations with their accuracy and status
iterations = [
    (0,  0.1312, "KEEP",    "Baseline: QCircuit + Z expvals"),
    (2,  0.4656, "KEEP",    "|ψ|² probs → MLP"),
    (3,  0.4094, "DISCARD", "Re+Im features (overfits)"),
    (4,  0.3969, "DISCARD", "Qubit perm augmentation"),
    (5,  0.6875, "KEEP",    "Subsystem purities + CE"),
    (6,  0.6531, "DISCARD", "Full-batch training"),
    (7,  0.7250, "KEEP",    "+1-qubit coherences"),
    (8,  0.7156, "DISCARD", "+2q Bell coherences"),
    (9,  0.5531, "DISCARD", "Phase-norm Re+Im"),
    (10, 0.7188, "DISCARD", "Multi-task learning"),
    (11, 0.7437, "KEEP",    "3-model ensemble"),
    (12, 0.7312, "DISCARD", "+VN entropy"),
    (13, 0.7000, "DISCARD", "Residual MLP"),
    (14, 0.5750, "DISCARD", "PQC approach"),
    (15, 0.7156, "DISCARD", "Cond. coherence diffs"),
    (16, 0.7375, "DISCARD", "SWA"),
    (17, 0.7312, "DISCARD", "Arch-diverse ensemble"),
    (18, 0.8281, "KEEP",    "DATA AUG: 500 extra/class"),
    (19, 0.8062, "DISCARD", "3000/class fast gen"),
    (20, 0.9187, "KEEP",    "10k/class, GPU (TARGET HIT!)"),
    (21, 0.9500, "KEEP",    "30k/class"),
    (22, 0.9750, "KEEP",    "100k/class, GPU"),
    (23, 0.9844, "KEEP",    "300k/class, 7-model ens."),
    (24, 0.9844, "KEEP",    "1M/class on-the-fly"),
]

iters    = [x[0] for x in iterations]
accs     = [x[1] for x in iterations]
statuses = [x[2] for x in iterations]
labels   = [x[3] for x in iterations]

kept_iters = [i for i, s in zip(iters, statuses) if s == "KEEP"]
kept_accs  = [a for a, s in zip(accs, statuses)  if s == "KEEP"]
disc_iters = [i for i, s in zip(iters, statuses) if s == "DISCARD"]
disc_accs  = [a for a, s in zip(accs, statuses)  if s == "DISCARD"]

# Running maximum of kept
running_best = []
best_so_far = 0.0
for i, (it, acc, st) in enumerate(zip(iters, accs, statuses)):
    if st == "KEEP":
        best_so_far = max(best_so_far, acc)
    running_best.append((it, best_so_far))

fig, ax = plt.subplots(figsize=(18, 9))

ax.scatter(disc_iters, disc_accs, c="#cccccc", s=30, alpha=0.6, zorder=2, label="Discarded")
ax.scatter(kept_iters, kept_accs, c="#2ecc71", s=80, zorder=4, label="Kept",
           edgecolors="black", linewidths=0.8)

rb_x = [r[0] for r in running_best]
rb_y = [r[1] for r in running_best]
ax.step(rb_x, rb_y, where="post", color="#27ae60", linewidth=2.5, alpha=0.8, zorder=3, label="Running best")

# Label kept experiments
for it, acc, label in zip(iters, accs, labels):
    if statuses[iters.index(it)] == "KEEP":
        ax.annotate(label, (it, acc), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color="#1a7a3a", alpha=0.9,
                    rotation=25, ha="left", va="bottom")

ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="90% target")
ax.axhline(y=0.125, color="gray", linestyle=":", alpha=0.5, linewidth=1.0, label="Random (12.5%)")

ax.set_xlabel("Experiment #", fontsize=13)
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("NTangled 8-Class Classification: 24 Experiments\nKey: data augmentation from circuit weights → 13.1% → 98.44%",
             fontsize=13)
ax.set_ylim(0.05, 1.05)
ax.set_xlim(-1, 26)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
