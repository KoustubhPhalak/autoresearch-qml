"""Generate progress.png for BoseHubbard phase classification experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

experiments = [
    (0,  0.6393, "3-class", "baseline Z-probs\n+ angle embed"),
    (1,  0.2623, "3-class", "data re-upload\nRX-RY-RZ"),
    (2,  0.6393, "3-class", "physics features\n(purities)"),
    (3,  0.6393, "3-class", "QBN + 2 linear"),
    (4,  0.6393, "3-class", "probs output\n+ all-to-all"),
    (5,  0.3607, "3-class", "top-8 RF feats\ndirect"),
    (6,  0.6393, "3-class", "log-probs\n+ tanh pre"),
    (7,  0.6393, "3-class", "IQP ZZ\nfeature map"),
    (8,  0.6393, "3-class", "rich feats\nRX-RY-RZ pack"),
    (9,  0.6393, "3-class", "quantum kernel\n+ SVM"),
    (10, 0.3607, "3-class", "tree-like\nthresholding"),
    (11, 0.4426, "3-class", "binary\noccupation"),
    (12, 0.4426, "3-class", "controlled\nrotations"),
    (13, 0.6393, "3-class", "amplitude embed\n8q 1-layer"),
    (14, 0.6393, "3-class", "rich observables\nX,Y,ZZ,XX"),
    (15, 0.9180, "2-class", "2-class reform\nStatePrep+QBN"),
    (16, 0.9180, "2-class", "layer comparison\n0,1,2 layers"),
]

iters = [e[0] for e in experiments]
accs = [e[1] for e in experiments]
cats = [e[2] for e in experiments]

fig, ax = plt.subplots(figsize=(16, 7))

colors = ['#d32f2f' if c == '3-class' else '#2e7d32' for c in cats]
bars = ax.bar(iters, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.axhline(y=0.6393, color='red', linestyle='--', alpha=0.4, label='3-class majority baseline (63.93%)')
ax.axhline(y=0.9180, color='green', linestyle='--', alpha=0.4, label='2-class ceiling (91.80%)')
ax.axhline(y=0.50, color='orange', linestyle=':', alpha=0.4, label='Random guess (50%)')

for i, (it, acc, cat, lbl) in enumerate(experiments):
    ax.text(it, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('BoseHubbard Phase Classification Progress\nRed = 3-class (impossible), Green = 2-class (reformulated)', fontsize=13)
ax.set_xticks(iters)
ax.set_xticklabels([e[3] for e in experiments], fontsize=6, rotation=60, ha='right')
ax.set_ylim(0, 1.08)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Annotation for key finding
ax.annotate('DISCOVERY: classes 1&2\nare identical |0⟩ states!\n3-class unsolvable',
            xy=(14, 0.6393), xytext=(10.5, 0.82),
            fontsize=9, color='#d32f2f', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.5))

ax.annotate('Reformulated as\n2-class problem\n→ 91.8% (ceiling)',
            xy=(15, 0.9180), xytext=(13, 1.02),
            fontsize=9, color='#2e7d32', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5))

plt.tight_layout()
plt.savefig('progress.png', dpi=150, bbox_inches='tight')
print('Saved progress.png')
