#!/usr/bin/env python3
"""
plot_progress.py — Visualization and Analysis of Pipeline Results
=================================================================

Reads results from results/ (produced by train.py) and baselines from
benchmarks/ (produced by prepare.py). Generates:

1. T-count comparison bar chart (pipeline vs baselines per benchmark)
2. Win/loss/tie summary
3. Per-stage ablation chart
4. Per-family breakdown
5. Scalability plot (T-count reduction vs circuit size)

Usage:
    python plot_progress.py [--results-dir results] [--benchmarks-dir benchmarks]
                            [--plots-dir plots]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Consistent color scheme
COLORS = {
    "original": "#cccccc",
    "full_reduce": "#6baed6",
    "full_optimize": "#2171b5",
    "phase_block": "#08519c",
    "pipeline": "#e6550d",
    "s1": "#fdae6b",
    "s2": "#e6550d",
    "s3": "#a63603",
    "win": "#2ca02c",
    "tie": "#ffbb78",
    "loss": "#d62728",
}


def load_results(results_dir: Path, benchmarks_dir: Path) -> dict:
    """Load pipeline results and baselines."""
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    summary_path = results_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return results, summary


def plot_tcount_comparison(results: dict, output_path: Path):
    """Bar chart comparing T-counts: original, baselines, pipeline."""
    # Collect data
    names = []
    original_t = []
    fr_t = []
    fo_t = []
    pipeline_t = []

    for name, entry in sorted(results.items()):
        if not entry.get("pipeline_result", {}).get("success", False):
            continue

        baselines = entry.get("baselines", {})
        orig = baselines.get("original", {}).get("t_count", 0)
        if orig == 0:
            continue

        names.append(name.split("/")[-1] if "/" in name else name)
        original_t.append(orig)
        fr_t.append(baselines.get("full_reduce", {}).get("t_count", orig))
        fo_t.append(baselines.get("full_optimize", {}).get("t_count", orig))
        pipeline_t.append(entry["pipeline_result"]["final"]["t_count"])

    if not names:
        print("No valid results to plot.")
        return

    # Limit to at most 30 benchmarks for readability
    if len(names) > 30:
        indices = np.linspace(0, len(names) - 1, 30, dtype=int)
        names = [names[i] for i in indices]
        original_t = [original_t[i] for i in indices]
        fr_t = [fr_t[i] for i in indices]
        fo_t = [fo_t[i] for i in indices]
        pipeline_t = [pipeline_t[i] for i in indices]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 7))
    ax.bar(x - 1.5 * width, original_t, width, label="Original", color=COLORS["original"])
    ax.bar(x - 0.5 * width, fr_t, width, label="full_reduce", color=COLORS["full_reduce"])
    ax.bar(x + 0.5 * width, fo_t, width, label="full_optimize", color=COLORS["full_optimize"])
    ax.bar(x + 1.5 * width, pipeline_t, width, label="Our Pipeline", color=COLORS["pipeline"])

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("T-count")
    ax.set_title("T-Count Comparison: Pipeline vs. Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_win_loss_tie(results: dict, output_path: Path):
    """Pie/bar chart of win/loss/tie vs each baseline."""
    baselines_to_compare = ["full_reduce", "full_optimize", "phase_block"]

    counts = {b: {"win": 0, "tie": 0, "loss": 0,
                   "unverified": 0, "error": 0} for b in baselines_to_compare}

    for name, entry in results.items():
        pr = entry.get("pipeline_result", {})
        baselines = entry.get("baselines", {})

        for bname in baselines_to_compare:
            bdata = baselines.get(bname, {})
            if not bdata.get("success", False):
                counts[bname]["error"] += 1
                continue
            if not pr.get("success", False):
                counts[bname]["error"] += 1
                continue

            verified = pr.get("final", {}).get("verified", "unverified")
            if verified != "verified":
                counts[bname]["unverified"] += 1
                continue

            our_t = pr["final"]["t_count"]
            b_t = bdata["t_count"]
            if our_t < b_t:
                counts[bname]["win"] += 1
            elif our_t == b_t:
                counts[bname]["tie"] += 1
            else:
                counts[bname]["loss"] += 1

    fig, axes = plt.subplots(1, len(baselines_to_compare),
                              figsize=(5 * len(baselines_to_compare), 4))
    if len(baselines_to_compare) == 1:
        axes = [axes]

    for ax, bname in zip(axes, baselines_to_compare):
        c = counts[bname]
        n_verified = c["win"] + c["tie"] + c["loss"]
        n_total = n_verified + c["unverified"] + c["error"]
        if n_total == 0:
            ax.set_title(f"vs {bname}\n(no data)")
            continue

        labels = ["Win", "Tie", "Loss", "Unverif", "Error"]
        values = [c["win"], c["tie"], c["loss"], c["unverified"], c["error"]]
        colors = [COLORS["win"], COLORS["tie"], COLORS["loss"], "#bbbbbb", "#999999"]

        bars = ax.bar(labels, values, color=colors)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val}\n({val/n_total*100:.0f}%)", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"vs {bname}\n(verified={n_verified}, total={n_total})")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Pipeline Win/Tie/Loss vs. Baselines (verified only in Win/Tie/Loss)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_stage_ablation(results: dict, output_path: Path):
    """Stacked bar chart showing T-count reduction contributed by each stage."""
    names = []
    s1_delta = []
    s2_delta = []
    s3_delta = []
    remaining_t = []

    for name, entry in sorted(results.items()):
        pr = entry.get("pipeline_result", {})
        if not pr.get("success", False):
            continue

        orig_t = pr.get("original_t_count", 0)
        if orig_t == 0:
            continue

        stages = pr.get("stages", {})
        d1 = stages.get("s1_zx_precondition", {}).get("delta_from_original", 0)
        d2 = stages.get("s2_phase_poly", {}).get("delta_from_s1", 0)
        d3 = stages.get("s3_cleanup", {}).get("delta_from_s2", 0)
        final_t = pr["final"]["t_count"]

        names.append(name.split("/")[-1] if "/" in name else name)
        s1_delta.append(max(0, d1))
        s2_delta.append(max(0, d2))
        s3_delta.append(max(0, d3))
        remaining_t.append(final_t)

    if not names:
        print("No valid results for ablation plot.")
        return

    # Limit for readability
    if len(names) > 25:
        indices = np.linspace(0, len(names) - 1, 25, dtype=int)
        names = [names[i] for i in indices]
        s1_delta = [s1_delta[i] for i in indices]
        s2_delta = [s2_delta[i] for i in indices]
        s3_delta = [s3_delta[i] for i in indices]
        remaining_t = [remaining_t[i] for i in indices]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))
    ax.bar(x, remaining_t, label="Remaining T-count", color=COLORS["original"])
    ax.bar(x, s3_delta, bottom=remaining_t, label="Stage 3 (cleanup)", color=COLORS["s3"])
    bottom2 = [r + d for r, d in zip(remaining_t, s3_delta)]
    ax.bar(x, s2_delta, bottom=bottom2, label="Stage 2 (phase-poly)", color=COLORS["s2"])
    bottom3 = [b + d for b, d in zip(bottom2, s2_delta)]
    ax.bar(x, s1_delta, bottom=bottom3, label="Stage 1 (ZX precond.)", color=COLORS["s1"])

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("T-count")
    ax.set_title("Per-Stage T-Count Reduction (Ablation)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_family_breakdown(results: dict, output_path: Path):
    """Grouped summary by circuit family."""
    family_stats = {}

    for name, entry in results.items():
        family = entry.get("family", "unknown")
        pr = entry.get("pipeline_result", {})
        if not pr.get("success", False):
            continue

        baselines = entry.get("baselines", {})
        orig_t = baselines.get("original", {}).get("t_count", 0)
        if orig_t == 0:
            continue

        our_t = pr["final"]["t_count"]
        fo_t = baselines.get("full_optimize", {}).get("t_count", orig_t)

        if family not in family_stats:
            family_stats[family] = {
                "n": 0,
                "total_orig": 0,
                "total_ours": 0,
                "total_fo": 0,
                "wins": 0,
                "losses": 0,
            }

        fs = family_stats[family]
        fs["n"] += 1
        fs["total_orig"] += orig_t
        fs["total_ours"] += our_t
        fs["total_fo"] += fo_t
        if our_t < fo_t:
            fs["wins"] += 1
        elif our_t > fo_t:
            fs["losses"] += 1

    if not family_stats:
        print("No family data to plot.")
        return

    families = sorted(family_stats.keys())
    x = np.arange(len(families))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: average reduction rate
    our_pct = []
    fo_pct = []
    for fam in families:
        fs = family_stats[fam]
        our_pct.append((1 - fs["total_ours"] / fs["total_orig"]) * 100)
        fo_pct.append((1 - fs["total_fo"] / fs["total_orig"]) * 100)

    width = 0.35
    ax1.bar(x - width / 2, fo_pct, width, label="full_optimize", color=COLORS["full_optimize"])
    ax1.bar(x + width / 2, our_pct, width, label="Our Pipeline", color=COLORS["pipeline"])
    ax1.set_xlabel("Circuit Family")
    ax1.set_ylabel("T-count Reduction (%)")
    ax1.set_title("Average T-Count Reduction by Family")
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=30, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: win/loss counts
    win_counts = [family_stats[f]["wins"] for f in families]
    loss_counts = [family_stats[f]["losses"] for f in families]
    n_counts = [family_stats[f]["n"] for f in families]

    ax2.bar(x - width / 2, win_counts, width, label="Wins vs full_optimize", color=COLORS["win"])
    ax2.bar(x + width / 2, loss_counts, width, label="Losses vs full_optimize", color=COLORS["loss"])
    for i, n in enumerate(n_counts):
        ax2.text(i, max(win_counts[i], loss_counts[i]) + 0.3, f"n={n}",
                 ha="center", fontsize=8)
    ax2.set_xlabel("Circuit Family")
    ax2.set_ylabel("Count")
    ax2.set_title("Wins/Losses vs full_optimize by Family")
    ax2.set_xticks(x)
    ax2.set_xticklabels(families, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scalability(results: dict, output_path: Path):
    """Scatter plot: T-count reduction vs original T-count (by qubit count)."""
    data_points = []

    for name, entry in results.items():
        pr = entry.get("pipeline_result", {})
        if not pr.get("success", False):
            continue

        baselines = entry.get("baselines", {})
        orig_t = baselines.get("original", {}).get("t_count", 0)
        if orig_t == 0:
            continue

        our_t = pr["final"]["t_count"]
        fo_t = baselines.get("full_optimize", {}).get("t_count", orig_t)
        qubits = entry.get("qubits", 0)
        total_time = pr["final"].get("total_time_s", 0)

        data_points.append({
            "orig_t": orig_t,
            "our_t": our_t,
            "fo_t": fo_t,
            "qubits": qubits,
            "our_reduction": (orig_t - our_t) / orig_t * 100 if orig_t > 0 else 0,
            "fo_reduction": (orig_t - fo_t) / orig_t * 100 if orig_t > 0 else 0,
            "time_s": total_time,
        })

    if not data_points:
        print("No data for scalability plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: reduction % vs original T-count
    orig_ts = [d["orig_t"] for d in data_points]
    our_reds = [d["our_reduction"] for d in data_points]
    fo_reds = [d["fo_reduction"] for d in data_points]
    qubits = [d["qubits"] for d in data_points]

    scatter1 = ax1.scatter(orig_ts, our_reds, c=qubits, cmap="viridis",
                            s=50, alpha=0.7, edgecolors="black", linewidths=0.5,
                            label="Our Pipeline")
    ax1.scatter(orig_ts, fo_reds, c="none", edgecolors=COLORS["full_optimize"],
                s=50, alpha=0.5, linewidths=1.5, marker="^", label="full_optimize")
    ax1.set_xlabel("Original T-count")
    ax1.set_ylabel("T-count Reduction (%)")
    ax1.set_title("Reduction vs. Circuit Size")
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Qubits")

    # Right: pipeline time vs original T-count
    times = [d["time_s"] for d in data_points]
    ax2.scatter(orig_ts, times, c=qubits, cmap="viridis",
                s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
    ax2.set_xlabel("Original T-count")
    ax2.set_ylabel("Pipeline Time (s)")
    ax2.set_title("Runtime Scalability")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary_table(results: dict, summary: dict):
    """Print a text summary table to console, including ALL benchmarks."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Benchmark':<30} {'Orig':>6} {'FR':>6} {'FO':>6} {'Ours':>6} {'vs FO':>8} {'Verif':>10}")
    print("-" * 80)

    n_failed = 0
    n_unverified = 0

    for name, entry in sorted(results.items()):
        pr = entry.get("pipeline_result", {})
        baselines = entry.get("baselines", {})
        orig_t = baselines.get("original", {}).get("t_count", "—")
        fr_t = baselines.get("full_reduce", {}).get("t_count", "—")
        fo_t = baselines.get("full_optimize", {}).get("t_count", "—")

        short_name = name.split("/")[-1] if "/" in name else name

        if not pr.get("success", False):
            # Failed run — still show it
            error = entry.get("error", pr.get("final", {}).get("verified", "error"))
            our_t = "FAIL"
            vs_fo = "ERROR"
            verif = str(error)[:10]
            n_failed += 1
            print(f"{short_name:<30} {orig_t:>6} {fr_t:>6} {fo_t:>6} {'FAIL':>6} {'ERROR':>8} {verif:>10}")
            continue

        our_t = pr["final"]["t_count"]
        verified = pr["final"].get("verified", "unverified")

        if isinstance(fo_t, int):
            raw_vs = "WIN" if our_t < fo_t else ("TIE" if our_t == fo_t else "LOSS")
            # Asterisk marks unverified — these are NOT included in official counts
            vs_fo = f"{raw_vs}*" if verified != "verified" else raw_vs
        else:
            vs_fo = "—"

        if verified == "unverified":
            n_unverified += 1
            verif_str = "unverif"
        elif verified == "verified":
            verif_str = "OK"
        elif verified == "failed":
            verif_str = "FAIL"
        else:
            verif_str = str(verified)[:10]

        print(f"{short_name:<30} {orig_t:>6} {fr_t:>6} {fo_t:>6} {our_t:>6} {vs_fo:>8} {verif_str:>10}")

    print("-" * 80)
    n_total = len(results)
    n_verified_total = n_total - n_failed - n_unverified
    print(f"Total: {n_total} benchmarks  |  Verified: {n_verified_total}  |  "
          f"Unverified: {n_unverified}  |  Failed: {n_failed}")
    if summary:
        n_v = summary.get('verified', n_verified_total)
        print(f"Successful: {summary.get('successful', '?')}/{summary.get('total', '?')}")
        print(f"Win/Tie/Loss vs full_optimize (verified only, n={n_v}):")
        print(f"  Wins:   {summary.get('wins_vs_full_optimize', '?')}/{n_v}")
        print(f"  Ties:   {summary.get('ties_vs_full_optimize', '?')}/{n_v}")
        print(f"  Losses: {summary.get('losses_vs_full_optimize', '?')}/{n_v}")
        if summary.get('unverified', 0) > 0:
            print(f"Unverified would-be wins/ties/losses (NOT in counts above): "
                  f"{summary.get('unverified_wins_vs_full_optimize', 0)}/"
                  f"{summary.get('unverified_ties_vs_full_optimize', 0)}/"
                  f"{summary.get('unverified_losses_vs_full_optimize', 0)}")
        print(f"Verification failures: {summary.get('verification_failures', '?')}")
        print(f"Pipeline errors:       {summary.get('pipeline_errors', '?')}")


def plot_avg_tcount_vs_iter(results: dict, output_path: Path):
    """
    Line chart: average T-count vs iteration for our pipeline vs. all baselines.

    Baseline averages (constant lines, from baselines.json over 50 benchmarks):
      original=45.12, full_reduce=24.60, full_optimize=23.44, phase_block=23.36

    Our pipeline average is reconstructed from the iteration history:
      avg_our = (total_fo_T - cumulative_win_savings + active_loss_overhead) / 50
    where win savings = fo_T - our_T for each win, and loss overhead = our_T - fo_T
    for iterations (iter0, iter6) that still had losses.
    """
    # Constant baseline averages (computed from baselines.json, 50 benchmarks)
    avg_original    = 45.12
    avg_fr          = 24.60
    avg_fo          = 23.44
    avg_pbo         = 23.36
    total_fo        = 1172   # sum of full_optimize T-counts across all 50 benchmarks

    # Per-iteration reconstruction of our pipeline's average T-count.
    # For each entry: (display_label, avg_our_T, status)
    # avg_our_T = None for DISCARD iters (not plotted on pipeline line).
    # Loss overhead at iter0: 12 losing circuits at full_reduce level (overhead=+62)
    # Loss overhead at iter6: 7 losing circuits (overhead=+27)
    # From iter10 onwards: 0 losses, all non-wins are ties at fo_T.
    iter_data = [
        # (label, avg_our,   status,      note)
        ("iter0",   24.60,  "DISCARD",   "2W 12L; losses at full_reduce level"),
        ("iter1",   None,   "DISCARD",   "verification bugs"),
        ("iter2",   24.60,  "DISCARD",   "12 verify fails"),
        ("iter3",   23.36,  "KEEP",      "2W 0L; losses→unverified"),
        ("iter4",   23.36,  "KEEP",      "same"),
        ("iter5",   23.36,  "KEEP",      "same"),
        ("iter6",   23.80,  "BEST",      "5W 7L; pp+rand wins, tof losses remain"),
        ("iter7",   None,   "DISCARD",   "shuffling broke equiv"),
        ("iter8/9", 23.42,  "DISCARD",   "3W 0L; regression"),
        ("iter10",  23.26,  "KEEP",      "5W 0L; losses→ties via Path B"),
        ("iter11",  23.26,  "KEEP",      "5W"),
        ("iter12",  23.30,  "DISCARD",   "4W regression"),
        ("iter13",  23.22,  "BEST",      "6W; GRPO adds pp_10q_40l"),
        ("iter14",  23.22,  "KEEP",      "6W; GRPO contamination fixed"),
        ("iter15",  23.16,  "BEST",      "7W; ZX+pbo adds tof_ladder_5"),
        ("iter16",  23.16,  "KEEP",      "7W"),
        ("iter17",  23.14,  "BEST",      "8W; TODD→pbo adds tof_ladder_3"),
        ("iter18",  23.02,  "BEST",      "10W; statevec verify adds tof_inter_6,8"),
        ("iter19",  23.02,  "KEEP",      "10W"),
        ("iter20",  22.98,  "BEST",      "12W; GRPO→pbo adds tof_inter_3,4"),
        ("iter21",  22.98,  "KEEP",      "12W; Path I2 no verified gain"),
        ("iter22",  22.94,  "BEST",      "13W; edge proxy adds tof_inter_5"),
        ("iter23",  22.94,  "KEEP",      "13W"),
        ("iter24",  22.94,  "KEEP",      "13W; compute wall reached"),
        ("iter25",  22.94,  "KEEP",      "13W; final"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw constant baseline lines
    x_full = list(range(len(iter_data)))
    ax.axhline(avg_original, color=COLORS["original"],   linewidth=2.0,
               linestyle="-",  label=f"Original ({avg_original:.2f})")
    ax.axhline(avg_fr,       color=COLORS["full_reduce"], linewidth=2.0,
               linestyle="-",  label=f"full_reduce ({avg_fr:.2f})")
    ax.axhline(avg_fo,       color=COLORS["full_optimize"], linewidth=2.0,
               linestyle="-",  label=f"full_optimize ({avg_fo:.2f})")
    ax.axhline(avg_pbo,      color=COLORS["phase_block"],  linewidth=2.0,
               linestyle="--", label=f"phase_block_optimize ({avg_pbo:.2f})")

    # Draw our pipeline line (connect only valid points)
    pipe_x, pipe_y = [], []
    scatter_x, scatter_y, scatter_c = [], [], []
    for i, (label, val, status, _) in enumerate(iter_data):
        if val is None:
            continue
        scatter_x.append(i)
        scatter_y.append(val)
        pipe_x.append(i)
        pipe_y.append(val)
        if status == "BEST":
            scatter_c.append("#2ca02c")
        elif status == "DISCARD":
            scatter_c.append("#d62728")
        else:
            scatter_c.append("#ff7f0e")

    # Connect non-DISCARD points with a line
    kept_x = [scatter_x[i] for i, (_, _, s, _) in
               enumerate([(l,v,st,n) for l,v,st,n in iter_data if v is not None])
               if s != "DISCARD"]
    kept_y = [scatter_y[i] for i, (_, _, s, _) in
               enumerate([(l,v,st,n) for l,v,st,n in iter_data if v is not None])
               if s != "DISCARD"]

    ax.plot(kept_x, kept_y, color="#ff7f0e", linewidth=2.5, zorder=3,
            label="Our pipeline (kept/best)")
    ax.scatter(scatter_x, scatter_y, c=scatter_c, s=70, zorder=5, edgecolors="white",
               linewidths=0.5)

    # Annotate best points
    for i, (label, val, status, _) in enumerate(iter_data):
        if val is not None and status == "BEST":
            ax.annotate(f"{val:.2f}", (i, val),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="#2ca02c", fontweight="bold")

    # Axis formatting — zoom into the competitive range (22–26.5).
    # Original (45.12) is labeled separately via text annotation.
    ax.set_xlim(-0.5, len(iter_data) - 0.5)
    ax.set_ylim(22.0, 26.5)
    # Annotate original T-count with an arrow pointing off the top
    ax.annotate(f"Original avg T = {avg_original:.2f}  ↑",
                xy=(len(iter_data) * 0.5, 26.3),
                fontsize=9, ha="center", color=COLORS["original"],
                fontweight="bold")
    ax.set_xticks(range(len(iter_data)))
    ax.set_xticklabels([d[0] for d in iter_data], rotation=40, ha="right", fontsize=7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average T-count (50 benchmarks)")
    ax.set_title("Average T-Count vs Iteration\n"
                 "Our Pipeline vs. Baselines (full_optimize is primary target)")
    ax.grid(axis="y", alpha=0.3)

    # Custom legend entries
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_els = [
        Line2D([0], [0], color=COLORS["original"],     linewidth=2, label=f"Original ({avg_original:.2f})"),
        Line2D([0], [0], color=COLORS["full_reduce"],  linewidth=2, label=f"full_reduce ({avg_fr:.2f})"),
        Line2D([0], [0], color=COLORS["full_optimize"],linewidth=2, label=f"full_optimize ({avg_fo:.2f})"),
        Line2D([0], [0], color=COLORS["phase_block"],  linewidth=2, linestyle="--",
               label=f"phase_block ({avg_pbo:.2f})"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2.5, label="Our pipeline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=8, label="Best result"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=8, label="Discarded"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_iter_progress(output_path: Path):
    """Line chart showing TEST_METRIC across iterations."""
    # Iteration history — update manually after each run
    iters = [
        ("iter0", 4.0, "BASELINE"),
        ("iter1", None, "DISCARD"),
        ("iter2", 4.0, "DISCARD"),
        ("iter3", 4.0, "KEEP"),
        ("iter4", 4.0, "KEEP"),
        ("iter5", 4.0, "KEEP"),
        ("iter6", 10.0, "BEST"),
        ("iter7", None, "DISCARD"),
        ("iter8/9", 6.0, "DISCARD"),
        ("iter10", 10.0, "KEEP"),
        ("iter11", 10.0, "KEEP"),
        ("iter12", 8.0, "DISCARD"),
        ("iter13", 12.0, "BEST"),
        ("iter14", 12.0, "KEEP"),
        ("iter15", 14.0, "BEST"),
        ("iter16", 14.0, "KEEP"),
        ("iter17", 16.0, "BEST"),
        ("iter18", 20.0, "BEST"),
        ("iter19", 20.0, "KEEP"),
        ("iter20", 24.0, "BEST"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    valid_x = []
    valid_y = []
    colors = []
    labels = []

    for i, (name, metric, status) in enumerate(iters):
        if metric is None:
            continue
        valid_x.append(i)
        valid_y.append(metric)
        if status == "BEST":
            colors.append("#2ca02c")
        elif status == "DISCARD":
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")
        labels.append(name)

    # Plot line of kept/best results
    kept_x = [valid_x[i] for i, (_, _, s) in enumerate(
        [(n, m, s) for n, m, s in iters if m is not None]) if s != "DISCARD"]
    kept_y = [valid_y[i] for i, (_, _, s) in enumerate(
        [(n, m, s) for n, m, s in iters if m is not None]) if s != "DISCARD"]

    # Draw running best line
    best_so_far = []
    current_best = 0
    for _, metric, status in iters:
        if metric is not None:
            if status != "DISCARD":
                current_best = max(current_best, metric)
            best_so_far.append(current_best)
        else:
            best_so_far.append(None)

    # Plot scatter for all valid points
    ax.scatter(valid_x, valid_y, c=colors, s=80, zorder=5)
    for x, y, label in zip(valid_x, valid_y, labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

    # Plot best-so-far line
    best_x = [i for i, v in enumerate(best_so_far) if v is not None]
    best_y = [v for v in best_so_far if v is not None]
    ax.step(best_x, best_y, where="post", color="#2ca02c", alpha=0.5,
            linewidth=2, linestyle="--", label="Running best")

    # Reference lines
    ax.axhline(y=10.0, color="orange", linestyle=":", alpha=0.7, label="iter6 best (10%)")
    ax.axhline(y=24.0, color="#2ca02c", linestyle=":", alpha=0.7, label="iter20 best (24%)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("TEST_METRIC (%)")
    ax.set_title("T-Count Optimization Pipeline: Progress Over Iterations\n"
                 "(TEST_METRIC = verified wins vs full_optimize / total benchmarks)")
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels([n for n, _, _ in iters], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="BEST"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="KEEP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="DISCARD"),
    ]
    ax.legend(handles=legend_elements + [
        Line2D([0], [0], color="#2ca02c", alpha=0.5, linewidth=2, linestyle="--", label="Running best"),
    ], fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot pipeline results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--plots-dir", type=str, default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmarks_dir = Path(args.benchmarks_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results, summary = load_results(results_dir, benchmarks_dir)
    print(f"Loaded {len(results)} result entries.\n")

    # Print text summary
    print_summary_table(results, summary)

    # Generate plots
    print("\nGenerating plots...")
    plot_tcount_comparison(results, plots_dir / "tcount_comparison.png")
    plot_win_loss_tie(results, plots_dir / "win_loss_tie.png")
    plot_per_stage_ablation(results, plots_dir / "per_stage_ablation.png")
    plot_per_family_breakdown(results, plots_dir / "per_family_breakdown.png")
    plot_scalability(results, plots_dir / "scalability.png")
    plot_iter_progress(plots_dir / "progress.png")
    plot_avg_tcount_vs_iter(results, plots_dir / "avg_tcount_vs_iter.png")

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()