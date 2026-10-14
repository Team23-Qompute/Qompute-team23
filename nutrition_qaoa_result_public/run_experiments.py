"""Brute-force ground truth, QUBO validation, and repeated QAOA experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from model import (
    BASE_DIR,
    NUTRIENTS,
    all_menus,
    bits_to_menus,
    build_quadratic_program,
    distinct_food_count,
    diversity_value,
    fit_nutrition_qubo,
    format_menu,
    load_food_data,
    load_policy,
    menus_to_bits,
    nutrition_cost,
    perturb_policy,
    policy_cost,
    rankdata,
    surrogate_policy_cost_bits,
    validate_surrogate,
)
from progress import ProgressLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "foods.csv")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--optimizers", nargs="+", default=["COBYLA", "Nelder-Mead"],
        choices=["COBYLA", "Powell", "Nelder-Mead"],
    )
    parser.add_argument("--max-evaluations", type=int, default=400,
                        help="Equal objective-evaluation budget for every depth and optimizer")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--shot-counts", nargs="+", type=int, default=[1024, 4096, 16384])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--qaoa-seeds", nargs="+", type=int, default=[1, 7, 21])
    parser.add_argument("--sensitivity-runs", type=int, default=100)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()
    validate_experiment_args(args)
    return args


def validate_experiment_args(args) -> None:
    if not args.depths or any(depth < 1 for depth in args.depths) or len(set(args.depths)) != len(args.depths):
        raise ValueError("depths must be unique positive integers")
    if args.restarts < 1 or args.max_evaluations < 1:
        raise ValueError("restarts and max-evaluations must be positive")
    if not args.qaoa_seeds or any(seed < 0 for seed in args.qaoa_seeds) or len(set(args.qaoa_seeds)) != len(args.qaoa_seeds):
        raise ValueError("qaoa-seeds must be nonnegative and unique")
    if not args.shot_counts or any(value < 1 for value in args.shot_counts):
        raise ValueError("shot-counts must be positive")
    if len(set(args.shot_counts)) != len(args.shot_counts) or args.shots not in args.shot_counts:
        raise ValueError("shot-counts must be unique and include --shots")
    if args.sensitivity_runs < 1:
        raise ValueError("sensitivity-runs must be positive")


def run_rng(seed: int, depth: int, optimizer_index: int, restart: int, stream: int):
    """Independent, order-invariant random streams for angles and shot draws."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, depth, optimizer_index, restart, stream])
    )


def optimizer_options(optimizer: str, evaluations: int) -> dict:
    # COBYLA's maxiter counts evaluations; scipy's simplex/direction methods
    # have a separate maxfev option. This keeps the comparison budget aligned.
    return {"maxiter": evaluations} if optimizer == "COBYLA" else {"maxfev": evaluations}


def strict_optimum_mask(costs: np.ndarray) -> np.ndarray:
    """Do not let np.isclose's default rtol count near-optimal states as exact."""
    return np.isclose(costs, np.min(costs), rtol=0.0, atol=1e-12)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def menu_record(index, menu, data, policy, total_cost=None) -> dict:
    totals = menu @ data.matrix
    nutrition = float(nutrition_cost(menu[None, :], data, policy)[0])
    diversity = float(diversity_value(menu[None, :], policy, len(data.foods))[0])
    return {
        "menu_index": int(index),
        "menu": format_menu(menu, data.foods),
        "serving_vector": menu.tolist(),
        "total_servings": int(menu.sum()),
        "distinct_foods": int(distinct_food_count(menu[None, :])[0]),
        "nutrition_cost": nutrition,
        "diversity_reward": diversity,
        "policy_cost": float(nutrition + diversity if total_cost is None else total_cost),
        "totals": {key: float(value) for key, value in zip(NUTRIENTS, totals)},
    }


def baseline_analysis(args, progress: ProgressLog | None = None):
    args.output.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.event("baseline_load_data", phase="baseline")
    data = load_food_data(args.data)
    policy = load_policy(args.config)
    menus = all_menus(len(data.foods), policy.max_servings_per_food)
    if progress:
        progress.event(
            "baseline_enumerated", phase="baseline", menus=len(menus),
            foods=len(data.foods),
        )
    nutrition_scores = nutrition_cost(menus, data, policy)
    true_scores = policy_cost(menus, data, policy)
    true_order = np.argsort(true_scores)
    true_best = int(true_order[0])

    if progress:
        progress.event("surrogate_fit_start", phase="surrogate_fit")
    surrogate = fit_nutrition_qubo(
        menus, nutrition_scores, policy.max_servings_per_food
    )
    validation = validate_surrogate(menus, true_scores, surrogate, policy)
    # A local weighted fit can reduce near-optimum error yet make the QUBO's
    # own optimum worse. Report both effects; do not silently switch models.
    fit_rows = [{"fit": "uniform", **validation}]
    cutoff = float(np.quantile(nutrition_scores, 0.01))
    for multiplier in (10, 100):
        weights = np.where(nutrition_scores <= cutoff, float(multiplier), 1.0)
        alternative = fit_nutrition_qubo(
            menus, nutrition_scores, policy.max_servings_per_food,
            sample_weights=weights,
        )
        fit_rows.append({
            "fit": f"lowest_1pct_nutrition_x{multiplier}",
            **validate_surrogate(menus, true_scores, alternative, policy),
        })
    write_csv(args.output / "surrogate_fit_comparison.csv", fit_rows)
    if progress:
        progress.event(
            "surrogate_fit_done", phase="baseline",
            spearman=validation["spearman"],
            top_1pct_overlap=validation["top_1pct_overlap"],
        )
    write_json(args.output / "surrogate_validation.json", validation)

    diversity_rows = []
    for reward in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]:
        changed = replace(policy, diversity_reward=reward)
        scores = policy_cost(menus, data, changed)
        index = int(np.argmin(scores))
        row = menu_record(index, menus[index], data, changed, scores[index])
        row = {"diversity_weight": reward, **row}
        del row["serving_vector"]
        del row["totals"]
        row["potassium_mg"] = float((menus[index] @ data.matrix)[4])
        row["sodium_mg"] = float((menus[index] @ data.matrix)[5])
        diversity_rows.append(row)
    write_csv(args.output / "diversity_sweep.csv", diversity_rows)

    rng = np.random.default_rng(args.seed)
    sensitivity_rows = []
    if progress:
        progress.event(
            "sensitivity_start", phase="sensitivity", runs=args.sensitivity_runs
        )
    for repetition in range(args.sensitivity_runs):
        changed = perturb_policy(policy, rng, 0.10)
        scores = policy_cost(menus, data, changed)
        index = int(np.argmin(scores))
        sensitivity_rows.append(
            {
                "repetition": repetition,
                "menu_index": index,
                "policy_cost": float(scores[index]),
                "total_servings": int(menus[index].sum()),
                "distinct_foods": int(distinct_food_count(menus[index:index+1])[0]),
                "menu": format_menu(menus[index], data.foods),
            }
        )
    write_csv(args.output / "sensitivity.csv", sensitivity_rows)
    frequencies = Counter(row["menu_index"] for row in sensitivity_rows)

    bits = menus_to_bits(menus, policy.max_servings_per_food)
    predicted = surrogate_policy_cost_bits(bits, surrogate, len(data.foods), policy)
    surrogate_best = int(np.argmin(predicted))
    summary = {
        "search_space": int(len(menus)),
        "fixed_total_servings": False,
        "true_best": menu_record(true_best, menus[true_best], data, policy),
        "surrogate_best": menu_record(
            surrogate_best, menus[surrogate_best], data, policy
        ),
        "surrogate_validation": validation,
        "sensitivity_unique_optima": len(frequencies),
        "sensitivity_baseline_optimum_frequency": int(frequencies.get(true_best, 0)),
    }
    write_json(args.output / "baseline_summary.json", summary)
    make_baseline_plot(
        args.output, true_scores, predicted, diversity_rows, sensitivity_rows
    )
    if progress:
        progress.event(
            "baseline_complete", phase="baseline_done", true_best_index=true_best,
            surrogate_best_true_rank=validation["surrogate_best_true_rank"],
        )
    return data, policy, menus, true_scores, surrogate, summary


def make_baseline_plot(output, true_scores, predicted, diversity_rows, sensitivity_rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].scatter(true_scores, predicted, s=7, alpha=0.18)
    low = min(true_scores.min(), predicted.min())
    high = max(true_scores.max(), predicted.max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="True policy cost", ylabel="QUBO cost", title="QUBO fidelity")

    weights = [row["diversity_weight"] for row in diversity_rows]
    axes[1].plot(weights, [row["distinct_foods"] for row in diversity_rows], "o-", label="types")
    axes[1].plot(weights, [row["total_servings"] for row in diversity_rows], "s-", label="servings")
    axes[1].set(xlabel="Diversity reward weight", title="Natural menu size")
    axes[1].legend()

    frequencies = Counter(row["menu_index"] for row in sensitivity_rows).most_common(10)
    axes[2].bar(range(len(frequencies)), [count for _, count in frequencies])
    axes[2].set_xticks(
        range(len(frequencies)), [str(index) for index, _ in frequencies],
        rotation=45, ha="right"
    )
    axes[2].set(xlabel="Menu index", ylabel="Times selected", title="±10% sensitivity")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "baseline_analysis.png", dpi=160)
    plt.close(fig)


def make_estimator(device):
    if device == "CPU":
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator()
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2
        if "GPU" not in AerSimulator().available_devices():
            raise RuntimeError("Qiskit Aer does not report a GPU device")
        return EstimatorV2(options={
            "backend_options": {
                "method": "statevector", "device": "GPU", "target_gpus": [0],
            },
        })
    except Exception as exc:
        raise RuntimeError("CUDA-compatible Qiskit Aer GPU is unavailable") from exc


def exact_probabilities(circuit, parameters, device):
    bound = circuit.assign_parameters(parameters)
    if device == "CPU":
        from qiskit.quantum_info import Statevector
        return Statevector(bound).probabilities()
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    simulator = AerSimulator(method="statevector", device="GPU", target_gpus=[0])
    saved = bound.copy()
    saved.save_statevector()
    compiled = transpile(saved, simulator)
    state = simulator.run(compiled).result().get_statevector(compiled)
    return np.abs(np.asarray(state)) ** 2


def run_qaoa(args, data, policy, true_scores, surrogate, progress: ProgressLog):
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz

    progress.event("qubo_build_start", phase="qubo_build")
    qp = build_quadratic_program(surrogate, data, policy)
    cost_operator, offset = qp.to_ising()
    scale = float(np.max(np.abs(cost_operator.coeffs)))
    hamiltonian = (cost_operator / scale).simplify()
    nq = cost_operator.num_qubits
    progress.event(
        "hamiltonian_built", phase="hamiltonian_validation",
        qubits=nq, pauli_terms=len(cost_operator), scale=scale,
    )
    indices = np.arange(2**nq, dtype=np.uint64)
    bits = ((indices[:, None] >> np.arange(nq, dtype=np.uint64)) & 1).astype(np.int8)
    basis_menus = bits_to_menus(
        bits, len(data.foods), policy.max_servings_per_food
    )
    expected_qubo = surrogate_policy_cost_bits(
        bits, surrogate, len(data.foods), policy
    )
    # Verify the optimization model and Pauli operator on representative states.
    # This catches variable ordering or coefficient errors before a long run.
    check_indices = np.linspace(0, len(bits) - 1, 33, dtype=int)
    for state_index in check_indices:
        state_bits = bits[state_index]
        qp_value = float(qp.objective.evaluate(state_bits))
        if not np.isclose(qp_value, expected_qubo[state_index], atol=1e-8):
            raise RuntimeError("QuadraticProgram does not match the QUBO surrogate")
        energy = float(offset)
        for pauli, coefficient in zip(cost_operator.paulis, cost_operator.coeffs):
            sign = -1 if int(np.dot(state_bits, pauli.z.astype(int))) % 2 else 1
            energy += float(coefficient.real) * sign
        if not np.isclose(energy, qp_value, atol=1e-8):
            raise RuntimeError("Ising Hamiltonian does not match the QUBO")
    progress.event("hamiltonian_validated", phase="qaoa_setup", checked_states=len(check_indices))
    basis_true = policy_cost(basis_menus, data, policy)
    basis_qubo = expected_qubo
    basis_nutrition = nutrition_cost(basis_menus, data, policy)
    servings = basis_menus.sum(axis=1)
    distinct = distinct_food_count(basis_menus)
    ranks = rankdata(basis_true) + 1
    optimum = basis_true.min()
    optimum_mask = strict_optimum_mask(basis_true)
    near_optimum_mask = basis_true <= optimum + 0.001
    qubo_optimum_mask = strict_optimum_mask(basis_qubo)
    top_count = max(1, int(0.01 * len(basis_true)))
    top_mask = ranks <= top_count

    estimator = make_estimator(args.device)
    progress.event("estimator_ready", phase="qaoa_setup", device=args.device)
    progress.event(
        "qaoa_plan", phase="qaoa_setup",
        depths=args.depths, optimizers=args.optimizers,
        seeds=args.qaoa_seeds, restarts=args.restarts,
        planned_runs=len(args.depths) * len(args.optimizers)
                     * len(args.qaoa_seeds) * args.restarts,
        max_evaluations_per_run=args.max_evaluations,
        shot_counts=sorted(args.shot_counts),
    )
    rows = []
    shot_rows = []
    for depth in args.depths:
        progress.event("circuit_build_start", phase="circuit_build", depth=depth)
        circuit = transpile(
            QAOAAnsatz(cost_operator=hamiltonian, reps=depth),
            basis_gates=["h", "rx", "rz", "rzz"], optimization_level=0,
        )
        progress.event(
            "circuit_ready", phase="qaoa_ready", depth=depth,
            parameters=circuit.num_parameters, gates=circuit.size(),
        )

        for optimizer_index, optimizer in enumerate(args.optimizers):
            for seed in args.qaoa_seeds:
              for restart in range(args.restarts):
                initial = run_rng(seed, depth, optimizer_index, restart, 0).uniform(
                    0, np.pi, circuit.num_parameters
                )
                evaluations = 0

                def objective(parameters):
                    nonlocal evaluations
                    evaluations += 1
                    progress.event(
                        "estimator_eval_start", phase="estimator", console=False,
                        depth=depth, optimizer=optimizer, seed=seed, restart=restart,
                        evaluation=evaluations,
                    )
                    result = estimator.run(
                        [(circuit, hamiltonian, parameters)]
                    ).result()[0]
                    value = float(result.data.evs)
                    progress.event(
                        "estimator_eval_done", phase="optimizer",
                        console=evaluations == 1 or evaluations % 10 == 0,
                        depth=depth, optimizer=optimizer, seed=seed, restart=restart,
                        evaluation=evaluations, energy=value,
                    )
                    return value

                progress.event(
                    "optimizer_start", phase="optimizer", depth=depth,
                    optimizer=optimizer, seed=seed, restart=restart,
                    max_evaluations=args.max_evaluations,
                )
                started = time.perf_counter()
                result = minimize(
                    objective, initial, method=optimizer,
                    options=optimizer_options(optimizer, args.max_evaluations),
                )
                elapsed = time.perf_counter() - started
                progress.event(
                    "optimizer_done", phase="probabilities", depth=depth,
                    optimizer=optimizer, seed=seed, restart=restart,
                    evaluations=int(result.nfev), seconds=round(elapsed, 2),
                    success=bool(result.success), message=str(result.message),
                )
                progress.event(
                    "probabilities_start", phase="probabilities", depth=depth,
                    optimizer=optimizer, seed=seed, restart=restart,
                )
                probabilities = exact_probabilities(circuit, result.x, args.device)
                progress.event(
                    "probabilities_done", phase="metrics", depth=depth,
                    optimizer=optimizer, seed=seed, restart=restart,
                )
                sampled = run_rng(seed, depth, optimizer_index, restart, 1).choice(
                    len(probabilities), max(args.shot_counts), p=probabilities
                )
                for shot_count in sorted(args.shot_counts):
                    current = sampled[:shot_count]
                    best = int(current[np.argmin(basis_true[current])])
                    if shot_count == args.shots:
                        primary_best = best
                    shot_rows.append({
                        "device": args.device, "depth": depth,
                        "optimizer": optimizer, "seed": seed,
                        "restart": restart, "shots": shot_count,
                        "true_optimum_observed": bool(np.any(optimum_mask[current])),
                        "true_top_1pct_observed": bool(np.any(top_mask[current])),
                        "best_sample_true_rank": int(ranks[best]),
                        "best_sample_policy_cost": float(basis_true[best]),
                        "best_sample_menu": format_menu(basis_menus[best], data.foods),
                    })
                rows.append({
                    "device": args.device,
                    "depth": depth,
                    "optimizer": optimizer,
                    "seed": seed,
                    "restart": restart,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                    "seconds": elapsed,
                    "expected_scaled_hamiltonian": float(result.fun),
                    "p_true_optimum": float(probabilities[optimum_mask].sum()),
                    "p_true_within_0p001": float(probabilities[near_optimum_mask].sum()),
                    "p_true_top_1pct": float(probabilities[top_mask].sum()),
                    "p_qubo_optimum": float(probabilities[qubo_optimum_mask].sum()),
                    "expected_qubo_policy_cost": float(probabilities @ basis_qubo),
                    "expected_true_policy_cost": float(probabilities @ basis_true),
                    "expected_nutrition_cost": float(probabilities @ basis_nutrition),
                    "expected_total_servings": float(probabilities @ servings),
                    "expected_distinct_foods": float(probabilities @ distinct),
                    "shots": args.shots,
                    "best_sample_true_rank": int(ranks[primary_best]),
                    "best_sample_policy_cost": float(basis_true[primary_best]),
                    "best_sample_total_servings": int(servings[primary_best]),
                    "best_sample_distinct_foods": int(distinct[primary_best]),
                    "best_sample_menu": format_menu(basis_menus[primary_best], data.foods),
                })
                progress.event(
                    "qaoa_run_complete", phase="qaoa_ready", depth=depth,
                    optimizer=optimizer, seed=seed, restart=restart,
                    p_true_top_1pct=rows[-1]["p_true_top_1pct"],
                    best_sample_true_rank=rows[-1]["best_sample_true_rank"],
                )
                write_csv(args.output / "qaoa_runs.csv", rows)
                write_csv(args.output / "qaoa_summary.csv", aggregate(rows))
                write_csv(args.output / "qaoa_seed_summary.csv", aggregate_seeds(rows))
                write_csv(args.output / "qaoa_shot_sweep.csv", shot_rows)
                write_csv(args.output / "qaoa_shot_sweep_summary.csv", aggregate_shots(shot_rows))
                progress.event(
                    "checkpoint_saved", phase="qaoa_ready", completed_runs=len(rows),
                )
    return rows, shot_rows


QAOA_METRICS = (
    "p_true_optimum", "p_true_within_0p001", "p_true_top_1pct",
    "p_qubo_optimum", "expected_qubo_policy_cost",
    "expected_true_policy_cost", "expected_nutrition_cost",
    "expected_total_servings", "expected_distinct_foods",
    "best_sample_true_rank", "seconds",
)


def aggregate_seeds(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["depth"], row["optimizer"], row["seed"]), []).append(row)
    output = []
    for (depth, optimizer, seed), members in sorted(groups.items()):
        record = {
            "device": members[0]["device"],
            "depth": depth, "optimizer": optimizer, "seed": seed,
            "restarts_completed": len(members),
            "successful_runs": sum(row["success"] for row in members),
        }
        for metric in QAOA_METRICS:
            values = np.asarray([row[metric] for row in members], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        record["best_expected_true_policy_cost"] = float(min(
            row["expected_true_policy_cost"] for row in members
        ))
        record["best_p_true_top_1pct"] = float(max(
            row["p_true_top_1pct"] for row in members
        ))
        output.append(record)
    return output


def aggregate(rows):
    """Compare depths using independent seed means, not pooled restarts."""
    groups = {}
    for record in aggregate_seeds(rows):
        groups.setdefault((record["depth"], record["optimizer"]), []).append(record)
    output = []
    for (depth, optimizer), seeds in sorted(groups.items()):
        record = {
            "device": seeds[0]["device"], "depth": depth,
            "optimizer": optimizer, "seeds": len(seeds),
            "runs": sum(row["restarts_completed"] for row in seeds),
            "successful_runs": sum(row["successful_runs"] for row in seeds),
        }
        for metric in QAOA_METRICS:
            values = np.asarray([row[f"{metric}_mean"] for row in seeds], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(record)
    return output


def aggregate_shots(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["depth"], row["optimizer"], row["shots"]), []).append(row)
    output = []
    for (depth, optimizer, shots), members in sorted(groups.items()):
        output.append({
            "device": members[0]["device"], "depth": depth,
            "optimizer": optimizer, "shots": shots, "runs": len(members),
            "true_optimum_observed_fraction": float(np.mean([
                row["true_optimum_observed"] for row in members
            ])),
            "true_top_1pct_observed_fraction": float(np.mean([
                row["true_top_1pct_observed"] for row in members
            ])),
            "best_sample_true_rank_mean": float(np.mean([
                row["best_sample_true_rank"] for row in members
            ])),
            "best_sample_true_rank_median": float(np.median([
                row["best_sample_true_rank"] for row in members
            ])),
        })
    return output


def make_qaoa_plot(output, rows):
    import matplotlib.pyplot as plt
    summary = aggregate(rows)
    labels = [f"p={row['depth']}\n{row['optimizer']}" for row in summary]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, [row["p_true_top_1pct_mean"] for row in summary],
                yerr=[row["p_true_top_1pct_std"] for row in summary], capsize=3)
    axes[0].axhline(0.01, color="black", linestyle="--", label="uniform")
    axes[0].set_ylabel("Probability of true top 1%")
    axes[0].legend()
    axes[1].bar(x, [row["expected_distinct_foods_mean"] for row in summary],
                yerr=[row["expected_distinct_foods_std"] for row in summary], capsize=3)
    axes[1].set_ylabel("Expected distinct foods")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "qaoa_summary.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    progress = ProgressLog(args.output, args.heartbeat_seconds)
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    progress.start(arguments=arguments)
    try:
        data, policy, menus, true_scores, surrogate, summary = baseline_analysis(
            args, progress
        )
        print(json.dumps(summary, indent=2), flush=True)
        if not args.analysis_only:
            rows, shot_rows = run_qaoa(
                args, data, policy, true_scores, surrogate, progress
            )
            progress.event("results_write_start", phase="writing_results", runs=len(rows))
            write_csv(args.output / "qaoa_runs.csv", rows)
            write_csv(args.output / "qaoa_summary.csv", aggregate(rows))
            write_csv(args.output / "qaoa_seed_summary.csv", aggregate_seeds(rows))
            write_csv(args.output / "qaoa_shot_sweep.csv", shot_rows)
            write_csv(args.output / "qaoa_shot_sweep_summary.csv", aggregate_shots(shot_rows))
            make_qaoa_plot(args.output, rows)
        progress.event(
            "run_complete", phase="complete", state="complete",
            files=sorted(path.name for path in args.output.iterdir() if path.is_file()),
        )
    except KeyboardInterrupt:
        progress.event(
            "run_interrupted", phase="interrupted", state="interrupted",
            traceback=traceback.format_exc(),
        )
        raise
    except BaseException as exc:
        progress.event(
            "run_failed", phase="failed", state="failed",
            error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        progress.stop()


if __name__ == "__main__":
    main()
