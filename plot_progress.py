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
    (15, 0.9180, "2-class", "2-class reform\nopen+closed"),
    (16, 0.9180, "2-class", "layer\ncomparison"),
    (17, 0.9672, "2-class", "optimized\nU=1.60"),
    (18, 1.0000, "2-class-open", "open only\n(clean labels)"),
    (19, 0.9672, "2-class", "threshold\nsweep"),
    (20, 1.0000, "2-class-open", "open boundary\nconfirmed"),
    (21, 0.9672, "2-class", "9q periodicity\nencoding"),
    (22, 0.9836, "2-class", "vacuum overlap\nanalysis"),
    (23, 0.9672, "2-class", "3-model\nensemble"),
    (24, 0.9836, "2-class", "2L + Z/X obs\n+ weighted loss"),
    (25, 0.9836, "2-class", "5-run sweep\nbest seed"),
]

iters = [e[0] for e in experiments]
accs = [e[1] for e in experiments]
cats = [e[2] for e in experiments]

fig, ax = plt.subplots(figsize=(20, 7))

color_map = {'3-class': '#d32f2f', '2-class': '#1565c0', '2-class-open': '#2e7d32'}
colors = [color_map[c] for c in cats]
bars = ax.bar(iters, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.axhline(y=0.6393, color='red', linestyle='--', alpha=0.4, label='3-class majority (63.93%)')
ax.axhline(y=0.9672, color='blue', linestyle=':', alpha=0.4, label='Combined naive ceiling (96.72%)')
ax.axhline(y=0.9836, color='#1565c0', linestyle='--', alpha=0.4, label='Combined best (98.36%)')

for i, (it, acc, cat, lbl) in enumerate(experiments):
    ax.text(it, acc + 0.015, f'{acc:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('BoseHubbard Phase Classification: 26 Iterations\nRed=3-class (unsolvable), Blue=2-class combined, Green=open-only (100%)', fontsize=13)
ax.set_xticks(iters)
ax.set_xticklabels([e[3] for e in experiments], fontsize=5.5, rotation=60, ha='right')
ax.set_ylim(0, 1.08)
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)

ax.annotate('DISCOVERY: classes 1&2\nidentical |0⟩ states',
            xy=(14, 0.6393), xytext=(9.5, 0.82),
            fontsize=8, color='#d32f2f', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.2))

ax.annotate('100% open\nboundary',
            xy=(18, 1.0), xytext=(16.5, 1.05),
            fontsize=8, color='#2e7d32', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5))

ax.annotate('98.36% combined\n(60/61 correct)',
            xy=(25, 0.9836), xytext=(22.5, 1.04),
            fontsize=8, color='#1565c0', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5))

plt.tight_layout()
plt.savefig('progress.png', dpi=150, bbox_inches='tight')
print('Saved progress.png')
