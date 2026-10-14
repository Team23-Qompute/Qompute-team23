"""Tests for the unconstrained, diversity-reward nutrition model."""

import unittest
from dataclasses import replace

import numpy as np

from model import (
    _minimum,
    _range,
    _target,
    all_menus,
    bits_to_menus,
    build_quadratic_program,
    distinct_food_count,
    diversity_value,
    fit_nutrition_qubo,
    load_food_data,
    load_policy,
    menus_to_bits,
    nutrition_cost,
    policy_cost,
    surrogate_policy_cost_bits,
    validate_surrogate,
)


class PenaltyTests(unittest.TestCase):
    def test_target(self):
        np.testing.assert_allclose(
            _target(np.array([90, 100, 110]), 100), [0.01, 0, 0.01]
        )

    def test_range(self):
        np.testing.assert_allclose(
            _range(np.array([50, 75, 100]), 60, 90),
            [(10 / 60) ** 2, 0, (10 / 90) ** 2],
        )

    def test_minimum(self):
        np.testing.assert_allclose(
            _minimum(np.array([50, 100, 150]), 100), [0.25, 0, 0]
        )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_food_data()
        cls.policy = load_policy()
        cls.menus = all_menus(
            len(cls.data.foods), cls.policy.max_servings_per_food
        )
        cls.bits = menus_to_bits(cls.menus, cls.policy.max_servings_per_food)
        cls.nutrition = nutrition_cost(cls.menus, cls.data, cls.policy)
        cls.true = policy_cost(cls.menus, cls.data, cls.policy)
        cls.surrogate = fit_nutrition_qubo(
            cls.menus, cls.nutrition, cls.policy.max_servings_per_food
        )

    def test_encoding_round_trip(self):
        decoded = bits_to_menus(
            self.bits, len(self.data.foods), self.policy.max_servings_per_food
        )
        np.testing.assert_array_equal(decoded, self.menus)

    def test_brute_force_uses_all_menus(self):
        self.assertEqual(len(self.menus), 4 ** len(self.data.foods))
        best = self.menus[int(np.argmin(self.true))]
        self.assertGreater(int(best.sum()), 0)

    def test_diversity_rewards_types_not_repeated_servings(self):
        policy = replace(self.policy, diversity_reward=0.08)
        zero = np.zeros((1, len(self.data.foods)), dtype=int)
        one = zero.copy(); one[0, 0] = 1
        two = zero.copy(); two[0, 0] = 2
        three = zero.copy(); three[0, 0] = 3
        one_reward = diversity_value(one, policy, len(self.data.foods))[0]
        self.assertEqual(diversity_value(zero, policy, len(self.data.foods))[0], 0)
        self.assertAlmostEqual(one_reward, -0.01)
        self.assertAlmostEqual(
            diversity_value(two, policy, len(self.data.foods))[0], one_reward
        )
        self.assertAlmostEqual(
            diversity_value(three, policy, len(self.data.foods))[0], one_reward
        )

    def test_two_bit_or_is_exact_presence_reward(self):
        reward = self.policy.diversity_reward / len(self.data.foods)
        for low, high, portions in [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]:
            qubo_value = -reward * low - reward * high + reward * low * high
            menu = np.zeros((1, len(self.data.foods)), dtype=int)
            menu[0, 0] = portions
            direct = diversity_value(menu, self.policy, len(self.data.foods))[0]
            self.assertAlmostEqual(qubo_value, direct)

    def test_distinct_count(self):
        menu = np.array([[0, 3, 0, 1, 2, 0, 0, 0]])
        self.assertEqual(int(distinct_food_count(menu)[0]), 3)

    def test_policy_has_no_total_serving_penalty(self):
        # Policy cost must equal nutrition cost plus only the diversity reward.
        expected = self.nutrition + diversity_value(
            self.menus, self.policy, len(self.data.foods)
        )
        np.testing.assert_allclose(self.true, expected)

    def test_surrogate_metrics(self):
        report = validate_surrogate(
            self.menus, self.true, self.surrogate, self.policy
        )
        self.assertEqual(report["menus"], len(self.menus))
        self.assertTrue(np.isfinite(report["rmse"]))
        self.assertGreater(report["spearman"], 0.8)
        self.assertEqual(
            report["near_optimum_menus"],
            int(np.count_nonzero(self.true <= self.true.min() + 0.001)),
        )
        self.assertGreaterEqual(report["true_best_surrogate_rank"], 1)
        self.assertGreater(report["near_optimum_mae"], 0)

    def test_weighted_surrogate_is_diagnostic_and_validates_weights(self):
        with self.assertRaises(ValueError):
            fit_nutrition_qubo(
                self.menus[:16], self.nutrition[:16],
                self.policy.max_servings_per_food, sample_weights=np.zeros(16),
            )
        alternative = fit_nutrition_qubo(
            self.menus[:16], self.nutrition[:16],
            self.policy.max_servings_per_food,
            sample_weights=np.linspace(1, 2, 16),
        )
        self.assertTrue(np.all(np.isfinite(alternative.linear)))

    def test_qiskit_objective_matches_surrogate_when_available(self):
        try:
            qp = build_quadratic_program(self.surrogate, self.data, self.policy)
        except ModuleNotFoundError:
            self.skipTest("Qiskit Optimization is not installed")
        rng = np.random.default_rng(7)
        for _ in range(20):
            bits = rng.integers(0, 2, self.bits.shape[1])
            expected = float(
                surrogate_policy_cost_bits(
                    bits, self.surrogate, len(self.data.foods), self.policy
                )
            )
            self.assertAlmostEqual(float(qp.objective.evaluate(bits)), expected, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
