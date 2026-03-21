"""
Plot progress for Lattice Reduction Cost Model experiments.
Reads iteration_history.json, generates progress.png.
"""

import os, json, math
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


def record_iteration(score, r2_interp, r2_extrap, theory_corr, label="iter0",
                     extra_metrics=None):
    """Append iteration stats to history file."""
    history = load_history()
    entry = {
        "label": label,
        "pred_score": score,
        "r2_interp": r2_interp,
        "r2_extrap": r2_extrap,
        "theory_corr": theory_corr,
    }
    if extra_metrics:
        entry.update(extra_metrics)
    history.append(entry)
    save_history(history)
    return history


def save_progress(y_interp, y_interp_pred, y_extrap, y_extrap_pred,
                  score, history=None, meta_interp=None, meta_extrap=None,
                  filename="progress.png"):
    """Generate 2×3 diagnostic progress.png."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Top-left: Interpolation scatter ──
    ax = axes[0, 0]
    if meta_interp is not None:
        colors_i = ["steelblue" if m["attack"] == "primal" else "mediumseagreen"
                     for m in meta_interp]
    else:
        colors_i = "steelblue"
    ax.scatter(y_interp, y_interp_pred, alpha=0.5, s=18, c=colors_i,
               edgecolors="white", linewidths=0.3)
    lims = [min(y_interp.min(), y_interp_pred.min()) - 1,
            max(y_interp.max(), y_interp_pred.max()) + 1]
    ax.plot(lims, lims, "r--", lw=1, label="y = x")
    ss_res = np.sum((y_interp - y_interp_pred) ** 2)
    ss_tot = np.sum((y_interp - y_interp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    ax.set_xlabel("Actual log₂(time)")
    ax.set_ylabel("Predicted log₂(time)")
    ax.set_title(f"Interpolation  (R² = {r2:.4f})")
    ax.legend(fontsize=8)

    # ── Top-middle: Extrapolation scatter ──
    ax = axes[0, 1]
    if len(y_extrap) > 0:
        if meta_extrap is not None:
            colors_e = ["darkorange" if m["attack"] == "primal" else "orchid"
                         for m in meta_extrap]
        else:
            colors_e = "darkorange"
        ax.scatter(y_extrap, y_extrap_pred, alpha=0.5, s=18, c=colors_e,
                   edgecolors="white", linewidths=0.3)
        lims = [min(y_extrap.min(), y_extrap_pred.min()) - 1,
                max(y_extrap.max(), y_extrap_pred.max()) + 1]
        ax.plot(lims, lims, "r--", lw=1)
        ss_res = np.sum((y_extrap - y_extrap_pred) ** 2)
        ss_tot = np.sum((y_extrap - y_extrap.mean()) ** 2)
        r2e = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
        ax.set_title(f"Extrapolation  (R² = {r2e:.4f})")
    else:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center",
                fontsize=20, color="gray")
        ax.set_title("No extrapolation data")
    ax.set_xlabel("Actual log₂(time)")
    ax.set_ylabel("Predicted log₂(time)")

    # ── Top-right: Residuals vs β ──
    ax = axes[0, 2]
    if meta_interp is not None and meta_extrap is not None:
        betas_i = [m["beta"] for m in meta_interp]
        betas_e = [m["beta"] for m in meta_extrap]
        res_i = y_interp - y_interp_pred
        res_e = y_extrap - y_extrap_pred if len(y_extrap) > 0 else np.array([])
        ax.scatter(betas_i, res_i, alpha=0.4, s=12, color="steelblue", label="Interp")
        if len(res_e) > 0:
            ax.scatter(betas_e, res_e, alpha=0.4, s=12, color="darkorange", label="Extrap")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel("β (BKZ block size)")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title("Residuals vs Block Size")
        ax.legend(fontsize=8)
    else:
        all_res = np.concatenate([y_interp - y_interp_pred] +
                                  ([y_extrap - y_extrap_pred] if len(y_extrap) > 0 else []))
        ax.hist(all_res, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_xlabel("Residual")
        ax.set_title("Residual Distribution")

    # ── Bottom-left: PRED_SCORE history ──
    ax = axes[1, 0]
    if history and len(history) > 0:
        iters = range(len(history))
        scores = [h["pred_score"] for h in history]
        best_score = max(scores)
        best_idx = scores.index(best_score)
        ax.plot(list(iters), scores, "b-o", markersize=5, linewidth=1.5)
        ax.scatter([best_idx], [best_score], color="gold", s=100, zorder=5,
                   edgecolors="black", linewidths=1, label=f"Best = {best_score:.4f}")
        ax.axhline(best_score, color="green", ls=":", lw=1, alpha=0.5)
        labels = [h["label"] for h in history]
        ax.set_xticks(list(iters))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("PRED_SCORE")
        ax.set_title("PRED_SCORE Over Iterations")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(min(0, min(scores) - 0.05), min(1.05, max(scores) + 0.1))
    else:
        ax.set_title("No history yet")

    # ── Bottom-middle: Component scores ──
    ax = axes[1, 1]
    if history and len(history) > 0:
        iters = list(range(len(history)))
        labels = [h["label"] for h in history]
        ax.plot(iters, [h.get("r2_interp", 0) for h in history], "s-",
                color="steelblue", markersize=4, linewidth=1.2, label="R²_interp (×0.3)")
        ax.plot(iters, [h.get("r2_extrap", 0) for h in history], "D-",
                color="darkorange", markersize=4, linewidth=1.2, label="R²_extrap (×0.5)")
        ax.plot(iters, [h.get("theory_corr", 0) for h in history], "^-",
                color="green", markersize=4, linewidth=1.2, label="Theory corr (×0.2)")
        ax.set_xticks(iters)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Component Score")
        ax.set_title("Score Components")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.set_title("No history yet")

    # ── Bottom-right: Residuals vs dimension n ──
    ax = axes[1, 2]
    if meta_interp is not None and meta_extrap is not None:
        ns_i = np.array([m["n"] for m in meta_interp])
        ns_e = np.array([m["n"] for m in meta_extrap])
        res_i = y_interp - y_interp_pred
        res_e = y_extrap - y_extrap_pred if len(y_extrap) > 0 else np.array([])

        ax.scatter(ns_i, res_i, alpha=0.4, s=12, color="steelblue", label="Interp")
        if len(res_e) > 0:
            ax.scatter(ns_e, res_e, alpha=0.4, s=12, color="darkorange", label="Extrap")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel("LWE dimension n")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title("Residuals vs Dimension")
        ax.legend(fontsize=8)

        # Per-dim mean residual markers
        all_ns = np.concatenate([ns_i] + ([ns_e] if len(res_e) > 0 else []))
        all_res = np.concatenate([res_i] + ([res_e] if len(res_e) > 0 else []))
        for n_val in sorted(set(all_ns)):
            mask = all_ns == n_val
            if mask.sum() > 0:
                ax.plot(n_val, all_res[mask].mean(), "k_", markersize=15, markeredgewidth=2)
    else:
        ax.text(0.5, 0.5, "No metadata", transform=ax.transAxes, ha="center",
                fontsize=14, color="gray")
        ax.set_title("Residuals vs Dimension")

    fig.suptitle(f"Lattice Reduction Cost Model — PRED_SCORE = {score:.4f}  "
                 f"(primal + dual, small secrets)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=120)
    plt.close()
    print(f"  Saved {filename}")


if __name__ == "__main__":
    history = load_history()
    if history:
        latest = history[-1]
        print(f"Latest: {latest['label']}  PRED_SCORE={latest['pred_score']:.4f}")
        print(f"  R²_interp={latest.get('r2_interp', 0):.4f}  "
              f"R²_extrap={latest.get('r2_extrap', 0):.4f}  "
              f"theory_corr={latest.get('theory_corr', 0):.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        iters = list(range(len(history)))
        labels_list = [h["label"] for h in history]
        scores = [h["pred_score"] for h in history]
        r2i = [h.get("r2_interp", 0) for h in history]
        r2e = [h.get("r2_extrap", 0) for h in history]
        tc = [h.get("theory_corr", 0) for h in history]

        best = max(scores)
        colors = ["gold" if s == best else "#4CAF50" if s > 0.5 else "#2196F3" for s in scores]
        axes[0].bar(iters, scores, color=colors, alpha=0.8, edgecolor="white")
        axes[0].axhline(best, color="green", ls=":", lw=1, alpha=0.5)
        axes[0].set_xticks(iters)
        axes[0].set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("PRED_SCORE")
        axes[0].set_title(f"PRED_SCORE  (best = {best:.4f})")

        w_interp = [0.3 * v for v in r2i]
        w_extrap = [0.5 * v for v in r2e]
        w_theory = [0.2 * v for v in tc]
        axes[1].bar(iters, w_interp, color="steelblue", alpha=0.8, label="R²_interp (30%)")
        axes[1].bar(iters, w_extrap, bottom=w_interp, color="darkorange", alpha=0.8,
                    label="R²_extrap (50%)")
        bottoms = [a + b for a, b in zip(w_interp, w_extrap)]
        axes[1].bar(iters, w_theory, bottom=bottoms, color="green", alpha=0.8,
                    label="Theory corr (20%)")
        axes[1].set_xticks(iters)
        axes[1].set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Weighted Contribution")
        axes[1].set_title("Score Breakdown")
        axes[1].legend(fontsize=7)

        axes[2].plot(iters, r2i, "s-", color="steelblue", markersize=5, linewidth=1.5,
                     label="R²_interp")
        axes[2].plot(iters, r2e, "D-", color="darkorange", markersize=5, linewidth=1.5,
                     label="R²_extrap")
        axes[2].plot(iters, tc, "^-", color="green", markersize=5, linewidth=1.5,
                     label="Theory corr")
        axes[2].set_xticks(iters)
        axes[2].set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("Score")
        axes[2].set_title("Component Scores (Raw)")
        axes[2].legend(fontsize=8)
        axes[2].set_ylim(-0.1, 1.1)

        plt.suptitle("Lattice Reduction Cost Model — Experiment Progress",
                      fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("progress.png", dpi=120)
        plt.close()
        print("Saved progress.png")
    else:
        print("No history found. Run train.py first.")