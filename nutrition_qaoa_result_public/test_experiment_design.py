"""Tests for the multi-seed experiment design and honest reporting."""

import unittest
import ast
import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

from model import FoodData, NutritionPolicy, QuboSurrogate
from build_bundle import PROJECT
from run_experiments import (
    QAOA_METRICS,
    aggregate,
    aggregate_seeds,
    aggregate_shots,
    optimizer_options,
    parse_args,
    run_rng,
    run_qaoa,
    strict_optimum_mask,
    validate_experiment_args,
)


def valid_args():
    return Namespace(
        depths=[1, 2, 3, 4], restarts=3, max_evaluations=400,
        qaoa_seeds=[1, 7, 21], shot_counts=[1024, 4096, 16384],
        shots=4096, sensitivity_runs=100,
    )


class ExperimentDesignTests(unittest.TestCase):
    def test_default_design_includes_four_depths_and_independent_seeds(self):
        validate_experiment_args(valid_args())
        with patch.object(sys, "argv", ["run_experiments.py"]):
            defaults = parse_args()
        self.assertEqual(defaults.depths, [1, 2, 3, 4])
        self.assertEqual(defaults.qaoa_seeds, [1, 7, 21])
        self.assertEqual(defaults.restarts, 3)
        self.assertEqual(defaults.optimizers, ["COBYLA", "Nelder-Mead"])
        notebook = json.loads((PROJECT / "start_here.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        settings = next(source for source in cells if "RUN_QAOA = True" in source)
        assignments = {
            target.id: ast.literal_eval(statement.value)
            for statement in ast.parse(settings).body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name) and target.id in {
                "DEPTHS", "QAOA_SEEDS", "RESTARTS", "OPTIMIZERS", "SHOT_COUNTS"
            }
        }
        self.assertEqual(assignments["DEPTHS"], defaults.depths)
        self.assertEqual(assignments["QAOA_SEEDS"], defaults.qaoa_seeds)
        self.assertEqual(assignments["RESTARTS"], defaults.restarts)
        self.assertEqual(assignments["OPTIMIZERS"], defaults.optimizers)
        self.assertEqual(assignments["SHOT_COUNTS"], defaults.shot_counts)
        command = next(source for source in cells if "'--qaoa-seeds'" in source)
        self.assertIn("'--shot-counts'", command)
        self.assertIn("'--max-evaluations'", command)
        args = valid_args()
        args.qaoa_seeds = [7, 7]
        with self.assertRaises(ValueError):
            validate_experiment_args(args)

    def test_strict_optimum_does_not_count_near_best(self):
        mask = strict_optimum_mask(np.array([-0.02, -0.01999985207100592, 0.0]))
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_random_streams_reproduce_without_loop_order_dependence(self):
        first = run_rng(7, 2, 1, 0, 0).random(5)
        again = run_rng(7, 2, 1, 0, 0).random(5)
        other = run_rng(7, 2, 1, 0, 1).random(5)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))

    def test_optimizer_caps_are_function_evaluation_caps(self):
        self.assertEqual(optimizer_options("COBYLA", 400), {"maxiter": 400})
        self.assertEqual(optimizer_options("Nelder-Mead", 400), {"maxfev": 400})
        self.assertEqual(optimizer_options("Powell", 400), {"maxfev": 400})

    def test_summary_variability_uses_seed_means(self):
        rows = []
        for seed, values in [(1, [0.1, 0.3]), (7, [0.5, 0.7])]:
            for restart, value in enumerate(values):
                row = {
                    "device": "CPU", "depth": 2, "optimizer": "COBYLA",
                    "seed": seed, "restart": restart, "success": True,
                }
                row.update({metric: value for metric in QAOA_METRICS})
                rows.append(row)
        per_seed = aggregate_seeds(rows)
        self.assertEqual(len(per_seed), 2)
        self.assertAlmostEqual(per_seed[0]["p_true_top_1pct_mean"], 0.2)
        summary = aggregate(rows)[0]
        self.assertEqual((summary["seeds"], summary["runs"]), (2, 4))
        self.assertAlmostEqual(summary["p_true_top_1pct_mean"], 0.4)
        self.assertAlmostEqual(summary["p_true_top_1pct_std"], np.std([0.2, 0.6], ddof=1))
        self.assertIsNone(aggregate(rows[:2])[0]["p_true_top_1pct_std"])

    def test_shot_summary_reports_hit_rate(self):
        rows = [
            {"device": "CPU", "depth": 1, "optimizer": "COBYLA", "shots": 1024,
             "true_optimum_observed": hit, "true_top_1pct_observed": True,
             "best_sample_true_rank": rank}
            for hit, rank in [(True, 1), (False, 3)]
        ]
        summary = aggregate_shots(rows)[0]
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["true_optimum_observed_fraction"], 0.5)
        self.assertEqual(summary["best_sample_true_rank_mean"], 2.0)

    def test_qaoa_protocol_loops_and_checkpoints_without_real_qiskit(self):
        """Exercise p=4, seeds, restarts, optimizers, and shots with mocked physics."""
        fake_qiskit = ModuleType("qiskit")
        fake_circuit_module = ModuleType("qiskit.circuit")
        fake_library = ModuleType("qiskit.circuit.library")

        class FakeCircuit:
            def __init__(self, cost_operator, reps):
                self.num_parameters = 2 * reps
                self.reps = reps

            def size(self):
                return 10 * self.reps

        class FakeOperator:
            num_qubits = 2
            coeffs = np.array([1.0])
            paulis = [SimpleNamespace(z=np.zeros(2, dtype=int))]

            def __len__(self):
                return 1

            def __truediv__(self, scale):
                return self

            def simplify(self):
                return self

        class FakeQP:
            objective = SimpleNamespace(evaluate=lambda bits: 0.0)

            def to_ising(self):
                return FakeOperator(), -1.0

        class FakeEstimator:
            def run(self, jobs):
                return SimpleNamespace(result=lambda: [SimpleNamespace(
                    data=SimpleNamespace(evs=0.0)
                )])

        fake_qiskit.transpile = lambda circuit, *args, **kwargs: circuit
        fake_library.QAOAAnsatz = FakeCircuit
        data = FoodData(
            foods=("food",), matrix=np.array([[100, 10, 20, 1, 100, 100.0]]),
            sources=("test",), serving_grams=np.array([100.0]),
        )
        policy = NutritionPolicy(
            energy_target=200, protein_range=(10, 30), carbs_range=(20, 60),
            fat_range=(1, 10), potassium_minimum=100, sodium_range=(100, 300),
            weights=(1, 1, 1, 1, 1, 0.25, 1), diversity_reward=0,
            max_servings_per_food=3,
        )
        surrogate = QuboSurrogate(
            intercept=0.0, linear=np.zeros(2), pairs=((0, 1),),
            pair_coefficients=np.zeros(1),
        )
        fake_minimize = lambda objective, initial, **kwargs: SimpleNamespace(
            x=initial, fun=objective(initial), nfev=1, success=True, message="mock"
        )
        fake_progress = SimpleNamespace(event=lambda *args, **kwargs: None)
        with TemporaryDirectory() as temporary:
            args = Namespace(
                output=Path(temporary), depths=[1, 4], optimizers=["COBYLA", "Nelder-Mead"],
                qaoa_seeds=[1, 7], restarts=2, max_evaluations=400,
                shot_counts=[1, 4, 16], shots=4, device="CPU",
            )
            with patch.dict(sys.modules, {
                "qiskit": fake_qiskit, "qiskit.circuit": fake_circuit_module,
                "qiskit.circuit.library": fake_library,
            }), patch("run_experiments.build_quadratic_program", return_value=FakeQP()), \
                patch("run_experiments.make_estimator", return_value=FakeEstimator()), \
                patch("run_experiments.exact_probabilities", return_value=np.full(4, 0.25)), \
                patch("run_experiments.minimize", side_effect=fake_minimize):
                runs, shot_rows = run_qaoa(
                    args, data, policy, None, surrogate, fake_progress
                )
            self.assertEqual(len(runs), 16)
            self.assertEqual(len(shot_rows), 48)
            self.assertEqual({row["depth"] for row in runs}, {1, 4})
            self.assertEqual({row["seed"] for row in runs}, {1, 7})
            self.assertTrue((Path(temporary) / "qaoa_summary.csv").is_file())
            self.assertTrue((Path(temporary) / "qaoa_shot_sweep_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
