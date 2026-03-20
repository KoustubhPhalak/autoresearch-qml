"""
Plot progress for Quantum Circuit Compilation experiments.
Reads iteration_history.json and current results, generates progress.png.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_FILE = "iteration_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def record_iteration(results, test_acc, label):
    """Append this iteration's stats to the history file, segregated by qubit count and category."""
    history = load_history()

    # Per-qubit breakdown
    by_nq = {}
    for r in results:
        nq = r.get("nq", 3)  # default 3 for backward compat
        if nq not in by_nq:
            by_nq[nq] = {"beat": 0, "total": 0, "ours_cx": [], "qiskit_cx": []}
        by_nq[nq]["total"] += 1
        if r.get("beat"):
            by_nq[nq]["beat"] += 1
        if r.get("success"):
            by_nq[nq]["ours_cx"].append(r["ours"])
            by_nq[nq]["qiskit_cx"].append(r["qiskit"])

    # Per-category breakdown
    by_cat = {}
    for r in results:
        cat = r.get("category", "?")
        if cat not in by_cat:
            by_cat[cat] = {"beat": 0, "total": 0, "ours_cx": [], "qiskit_cx": []}
        by_cat[cat]["total"] += 1
        if r.get("beat"):
            by_cat[cat]["beat"] += 1
        if r.get("success"):
            by_cat[cat]["ours_cx"].append(r["ours"])
            by_cat[cat]["qiskit_cx"].append(r["qiskit"])

    # Build per-nq summary
    nq_summary = {}
    for nq, d in by_nq.items():
        nq_summary[str(nq)] = {
            "beat": d["beat"], "total": d["total"],
            "avg_ours": float(np.mean(d["ours_cx"])) if d["ours_cx"] else 0,
            "avg_qiskit": float(np.mean(d["qiskit_cx"])) if d["qiskit_cx"] else 0,
            "avg_saving": float(np.mean([q-o for o,q in zip(d["ours_cx"], d["qiskit_cx"])])) if d["ours_cx"] else 0,
        }

    successful = [r for r in results if r.get("success")]
    entry = {
        "label": label,
        "test_acc": test_acc,
        "n_eval": len(results),
        "n_beat": sum(1 for r in results if r.get("beat")),
        "avg_ours_cx": float(np.mean([r["ours"] for r in successful])) if successful else 0,
        "avg_qiskit_cx": float(np.mean([r["qiskit"] for r in successful])) if successful else 0,
        "avg_saving": float(np.mean([r["qiskit"] - r["ours"] for r in successful])) if successful else 0,
        "by_nq": nq_summary,
    }
    history.append(entry)
    save_history(history)
    return history


def save_progress(results, test_acc, history=None, filename="progress.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: per-category beat rate
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"beat": 0, "total": 0}
        cats[c]["total"] += 1
        if r["beat"]:
            cats[c]["beat"] += 1

    cat_names = sorted(cats.keys())
    rates = [cats[c]["beat"] / cats[c]["total"] if cats[c]["total"] > 0 else 0 for c in cat_names]
    colors = ["#4CAF50" if r > 0.5 else "#2196F3" if r > 0 else "#f44336" for r in rates]
    axes[0].bar(range(len(cat_names)), rates, color=colors)
    axes[0].set_xticks(range(len(cat_names)))
    axes[0].set_xticklabels(cat_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Beat Rate")
    axes[0].set_title(f"Beat rate by category\n(TEST_ACC={test_acc:.4f})")
    axes[0].set_ylim(0, 1.05)

    # Middle: average CX count comparison (Qiskit vs Ours) by category
    cats_cx = {}
    for r in results:
        if not r.get("success"):
            continue
        c = r["category"]
        if c not in cats_cx:
            cats_cx[c] = {"qiskit": [], "ours": []}
        cats_cx[c]["qiskit"].append(r["qiskit"])
        cats_cx[c]["ours"].append(r["ours"])

    cx_cats = sorted(cats_cx.keys())
    x = np.arange(len(cx_cats))
    width = 0.35
    qiskit_means = [np.mean(cats_cx[c]["qiskit"]) for c in cx_cats]
    ours_means = [np.mean(cats_cx[c]["ours"]) for c in cx_cats]
    axes[1].bar(x - width/2, qiskit_means, width, label="Qiskit opt-3 (static)", color="#f44336", alpha=0.8)
    axes[1].bar(x + width/2, ours_means, width, label="Ours", color="#4CAF50", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cx_cats, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Avg CX Gate Count")
    axes[1].set_title("CX Gate Count: Qiskit vs Ours")
    axes[1].legend()

    # Right: per-qubit CX savings over iterations
    if history and len(history) > 1:
        iters = list(range(len(history)))
        labels = [h["label"] for h in history]

        # Extract per-qubit savings
        for nq_str, color, marker in [("3", "#2196F3", "o"), ("4", "#FF9800", "s")]:
            savings_nq = []
            for h in history:
                by_nq = h.get("by_nq", {})
                if nq_str in by_nq:
                    savings_nq.append(by_nq[nq_str].get("avg_saving", 0))
                else:
                    savings_nq.append(0)
            if any(s > 0 for s in savings_nq):
                axes[2].plot(iters, savings_nq, f"-{marker}", color=color,
                            label=f"{nq_str}q avg CX saved", linewidth=2, markersize=6)

        ax2_twin = axes[2].twinx()
        accs = [h.get("test_acc", 0) for h in history]
        ax2_twin.plot(iters, accs, "k--o", label="TEST_ACC", linewidth=1, markersize=4, alpha=0.5)
        axes[2].set_xticks(iters)
        axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("Avg CX Gates Saved (per qubit count)")
        ax2_twin.set_ylabel("TEST_ACC", color="gray")
        ax2_twin.set_ylim(0, 1.05)
        axes[2].set_title("CX Savings by Qubit Count\n(Qiskit baselines are static)")
        axes[2].legend(loc="upper left", fontsize=7)
        ax2_twin.legend(loc="upper right", fontsize=7)
    else:
        our_cx = [r["ours"] for r in results if r.get("success")]
        bl_cx = [r["qiskit"] for r in results if r.get("success")]
        if our_cx:
            axes[2].scatter(bl_cx, our_cx,
                           c=["green" if o < b else "red" for o, b in zip(our_cx, bl_cx)],
                           alpha=0.6, s=30)
            mx = max(max(our_cx), max(bl_cx)) + 2
            axes[2].plot([0, mx], [0, mx], "k--", alpha=0.3, label="y=x (tie)")
            axes[2].set_xlabel("Qiskit opt-3 CX count")
            axes[2].set_ylabel("Our CX count")
            axes[2].set_title("Per-unitary CX comparison")
            axes[2].legend()

    plt.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close()


if __name__ == "__main__":
    history = load_history()
    if history:
        latest = history[-1]
        print(f"Latest: {latest['label']} TEST_ACC={latest['test_acc']:.4f}")
        fig, ax = plt.subplots(figsize=(10, 5))
        iters = range(len(history))
        savings = [h.get("avg_saving", 0) for h in history]
        accs = [h.get("test_acc", 0) for h in history]
        labels = [h["label"] for h in history]
        ax_twin = ax.twinx()
        ax.bar(iters, savings, color="#4CAF50", alpha=0.7, label="Avg CX saved")
        ax_twin.plot(iters, accs, "b-o", label="TEST_ACC", linewidth=2, markersize=6)
        ax.set_xticks(list(iters))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg CX Gates Saved vs Qiskit (static)")
        ax_twin.set_ylabel("TEST_ACC", color="blue")
        ax_twin.set_ylim(0, 1.05)
        ax.set_title("Circuit Compilation: Improvement Over Iterations")
        ax.legend(loc="upper left")
        ax_twin.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig("progress.png", dpi=120)
        plt.close()
        print("Saved progress.png")
    else:
        print("No history found")
