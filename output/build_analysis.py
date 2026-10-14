"""Rebuild privacy-safe QAOA summary figures from aggregate results.

Run without arguments to redraw the committed figures from analysis_data.json.
Pass --run-dir to refresh that JSON from a completed experiment directory first.
Only numeric aggregate fields are copied; logs and environment metadata stay out.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_FILE = OUTPUT_DIR / "analysis_data.json"
COLORS = {"COBYLA": "#206878", "Nelder-Mead": "#D47443"}
SEED_COLORS = {"1": "#206878", "7": "#D47443", "21": "#7658A4"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compact_summary(row: dict[str, str]) -> dict:
    fields = (
        "expected_true_policy_cost_mean",
        "expected_true_policy_cost_std",
        "expected_qubo_policy_cost_mean",
        "p_true_top_1pct_mean",
        "p_true_top_1pct_std",
        "p_true_optimum_mean",
        "p_true_within_0p001_mean",
    )
    result = {
        "depth": int(row["depth"]),
        "optimizer": row["optimizer"],
        "seeds": int(row["seeds"]),
        "runs": int(row["runs"]),
        "successful_runs": int(row["successful_runs"]),
    }
    result.update({field: float(row[field]) for field in fields})
    return result


def compact_seed(row: dict[str, str]) -> dict:
    return {
        "depth": int(row["depth"]),
        "optimizer": row["optimizer"],
        "seed": int(row["seed"]),
        "restarts_completed": int(row["restarts_completed"]),
        "expected_true_policy_cost_mean": float(row["expected_true_policy_cost_mean"]),
        "p_true_top_1pct_mean": float(row["p_true_top_1pct_mean"]),
    }


def build_data(run_dir: Path, exact_optima: int) -> dict:
    required = (
        "qaoa_summary.csv",
        "qaoa_seed_summary.csv",
        "qaoa_shot_sweep.csv",
        "surrogate_fit_comparison.csv",
        "baseline_summary.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {', '.join(missing)}")

    with (run_dir / "baseline_summary.json").open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    summary = [compact_summary(row) for row in csv_rows(run_dir / "qaoa_summary.csv")]
    seeds = [compact_seed(row) for row in csv_rows(run_dir / "qaoa_seed_summary.csv")]
    sweep = csv_rows(run_dir / "qaoa_shot_sweep.csv")
    shots = defaultdict(lambda: {"runs": 0, "exact_hits": 0})
    for row in sweep:
        count = int(row["shots"])
        shots[count]["runs"] += 1
        shots[count]["exact_hits"] += row["true_optimum_observed"].lower() == "true"
    fits = []
    for row in csv_rows(run_dir / "surrogate_fit_comparison.csv"):
        fits.append({
            "fit": row["fit"],
            "global_spearman": float(row["spearman"]),
            "near_optimum_spearman": float(row["near_optimum_spearman"]),
            "near_optimum_mae": float(row["near_optimum_mae"]),
            "near_optimum_menus": int(row["near_optimum_menus"]),
            "surrogate_best_true_rank": int(row["surrogate_best_true_rank"]),
        })
    payload = {
        "description": "Privacy-safe numeric summary of one completed CPU statevector study",
        "search_space": int(baseline["search_space"]),
        "exact_optima": exact_optima,
        "summary": summary,
        "seed_summary": seeds,
        "shot_observations": [
            {"shots": count, **values} for count, values in sorted(shots.items())
        ],
        "surrogate_fits": fits,
    }
    if len(summary) != 8 or len(seeds) != 24:
        raise ValueError("Expected four depths, two optimizers, and three seeds")
    if {row["shots"] for row in payload["shot_observations"]} != {1024, 4096, 16384}:
        raise ValueError("Unexpected shot-sweep design")
    if exact_optima < 1 or exact_optima > payload["search_space"]:
        raise ValueError("exact_optima must be between 1 and search_space")
    return payload


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#607078",
        "axes.labelcolor": "#263238",
        "text.color": "#263238",
        "xtick.color": "#263238",
        "ytick.color": "#263238",
        "grid.color": "#DDE4E7",
        "grid.linewidth": 0.7,
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=190, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)


def depth_tradeoffs(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for optimizer in ("COBYLA", "Nelder-Mead"):
        rows = sorted((r for r in data["summary"] if r["optimizer"] == optimizer), key=lambda r: r["depth"])
        x = np.array([r["depth"] for r in rows])
        color = COLORS[optimizer]
        axes[0].errorbar(
            x, [r["expected_true_policy_cost_mean"] for r in rows],
            yerr=[r["expected_true_policy_cost_std"] for r in rows],
            color=color, marker="o", capsize=4, linewidth=2, label=optimizer,
        )
        axes[1].errorbar(
            x, [100 * r["p_true_top_1pct_mean"] for r in rows],
            yerr=[100 * r["p_true_top_1pct_std"] for r in rows],
            color=color, marker="o", capsize=4, linewidth=2,
        )
        offset = -0.18 if optimizer == "COBYLA" else 0.18
        bars = axes[2].bar(
            x + offset, [r["successful_runs"] / r["runs"] * 100 for r in rows],
            width=0.34, color=color, alpha=0.88,
        )
        for bar, row in zip(bars, rows):
            axes[2].annotate(
                f"{row['successful_runs']}/{row['runs']}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
            )
    axes[0].set_title("A. Average true cost | lower is better")
    axes[0].set_ylabel("Expected policy cost (unitless)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("B. Chance of a top-1% menu | higher is better")
    axes[1].set_ylabel("Probability per draw (%)")
    axes[1].axhline(1, color="#53636C", linestyle="--", linewidth=1, label="Uniform 1%")
    axes[1].set_ylim(bottom=0)
    axes[2].set_title("C. Optimizer converged within budget")
    axes[2].set_ylabel("Runs converged (%)")
    axes[2].set_ylim(0, 114)
    for ax in axes:
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("QAOA depth p")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Depth trade-off under a fixed 400-evaluation budget", fontsize=15, y=0.99)
    fig.text(0.5, 0.015, "Means +/- SD across 3 independent seed means; 3 restarts per seed; 400 function evaluations per run.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.16, wspace=0.31)
    save(fig, "depth_tradeoffs.png")


def paired_seeds(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    rows = [r for r in data["seed_summary"] if r["optimizer"] == "Nelder-Mead" and r["depth"] in (2, 4)]
    for seed in (1, 7, 21):
        pair = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["depth"])
        if len(pair) != 2:
            raise ValueError(f"Missing paired p=2/p=4 results for seed {seed}")
        color = SEED_COLORS[str(seed)]
        axes[0].plot([2, 4], [r["expected_true_policy_cost_mean"] for r in pair], "o-", color=color, linewidth=2.3, markersize=7, label=f"Seed {seed}")
        axes[1].plot([2, 4], [100 * r["p_true_top_1pct_mean"] for r in pair], "o-", color=color, linewidth=2.3, markersize=7)
    for ax, field, factor in (
        (axes[0], "expected_true_policy_cost_mean", 1),
        (axes[1], "p_true_top_1pct_mean", 100),
    ):
        means = [np.mean([r[field] * factor for r in rows if r["depth"] == depth]) for depth in (2, 4)]
        ax.plot([2, 4], means, "s--", color="#222E34", linewidth=2, markersize=6, label="3-seed mean")
        for depth, value in zip((2, 4), means):
            ax.annotate(f"{value:.3f}" if factor == 1 else f"{value:.2f}%", (depth, value), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=9, color="#222E34")
        ax.set_xlim(1.7, 4.3)
        ax.set_xticks([2, 4], ["p=2", "p=4"])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_title("A. True expected cost | lower is better")
    axes[0].set_ylabel("Expected policy cost (unitless)")
    axes[0].set_ylim(0.04, 0.34)
    axes[1].set_title("B. Top-1% probability | higher is better")
    axes[1].set_ylabel("Probability per draw (%)")
    axes[1].set_ylim(0, 8.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Seed-by-seed comparison: the mean hides different winners", fontsize=15, y=0.99)
    fig.text(0.5, 0.015, "Nelder-Mead; each seed point averages 3 restarts. Seed 1 drives the p=2 cost advantage.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.16, wspace=0.28)
    save(fig, "paired_seed_variability.png")


def sampling_vs_uniform(data: dict) -> None:
    observations = sorted(data["shot_observations"], key=lambda r: r["shots"])
    shots = [r["shots"] for r in observations]
    observed = [100 * r["exact_hits"] / r["runs"] for r in observations]
    uniform = [100 * (1 - (1 - data["exact_optima"] / data["search_space"]) ** count) for count in shots]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(shots))
    ax.plot(x, observed, "o-", color=COLORS["COBYLA"], linewidth=2.8, markersize=9, label="QAOA observed")
    ax.plot(x, uniform, "s--", color=COLORS["Nelder-Mead"], linewidth=2.4, markersize=7, label="Uniform random benchmark")
    for index, (row, value) in enumerate(zip(observations, observed)):
        ax.annotate(f"{row['exact_hits']}/{row['runs']}  ({value:.1f}%)", (index, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    for index, value in enumerate(uniform):
        ax.annotate(f"{value:.1f}%", (index, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{count:,}" for count in shots])
    ax.set_xlabel("Simulated draws per QAOA run")
    ax.set_ylabel("Runs observing the exact optimum (%)")
    ax.set_ylim(0, 62)
    ax.set_xlim(-0.18, len(x) - 0.82)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("More draws improve the chance of seeing the exact optimum", fontsize=15, y=0.98)
    fig.text(0.5, 0.015, "QAOA: 72 optimized distributions (nested draw counts). Uniform: analytic benchmark with 1 optimum among 65,536 menus.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.16)
    save(fig, "sampling_vs_uniform.png")


def surrogate_quality(data: dict) -> None:
    fits = data["surrogate_fits"]
    uniform = next(row for row in fits if row["fit"] == "uniform")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    bars = axes[0].bar(
        ["All 65,536 menus", f"Near optimum\n({uniform['near_optimum_menus']} menus)"],
        [uniform["global_spearman"], uniform["near_optimum_spearman"]],
        color=[COLORS["COBYLA"], COLORS["Nelder-Mead"]], width=0.55,
    )
    for bar in bars:
        axes[0].annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[0].set_title("A. QUBO vs true-cost rank agreement")
    axes[0].set_ylabel("Spearman correlation (1 = same ranking)")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    labels = {"uniform": "Uniform fit", "lowest_1pct_nutrition_x10": "Low-cost x10", "lowest_1pct_nutrition_x100": "Low-cost x100"}
    for index, row in enumerate(fits):
        axes[1].scatter(row["near_optimum_mae"], row["surrogate_best_true_rank"], s=100, color=["#206878", "#D47443", "#7658A4"][index], zorder=3)
        axes[1].annotate(labels.get(row["fit"], row["fit"]), (row["near_optimum_mae"], row["surrogate_best_true_rank"]), xytext=(7, -2), textcoords="offset points", va="center", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 0.11)
    axes[1].set_ylim(10, 4000)
    axes[1].set_yticks([10, 100, 1000], ["10", "100", "1,000"])
    axes[1].minorticks_off()
    axes[1].set_title("B. Lower local error did not improve the winner")
    axes[1].set_xlabel("Near-optimum mean absolute error (lower is better)")
    axes[1].set_ylabel("True rank of QUBO's best menu (lower is better)")
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)
    fig.suptitle("A high global fit score can hide poor ordering at the top", fontsize=15, y=0.99)
    fig.text(0.5, 0.015, "Near optimum means true cost within 0.001 of the exact minimum. Weighted fits are diagnostics, not the QAOA objective.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.2, wspace=0.35)
    save(fig, "surrogate_quality.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Completed result directory to refresh analysis_data.json")
    parser.add_argument("--exact-optima", type=int, default=1, help="Number of exact optimal menus (this run: 1)")
    args = parser.parse_args()
    if args.run_dir:
        data = build_data(args.run_dir, args.exact_optima)
        DATA_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        with DATA_FILE.open(encoding="utf-8") as handle:
            data = json.load(handle)
    style()
    depth_tradeoffs(data)
    paired_seeds(data)
    sampling_vs_uniform(data)
    surrogate_quality(data)
    print(f"Wrote four figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
