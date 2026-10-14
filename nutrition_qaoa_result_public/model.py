"""Range-aware nutrition objective and compact QUBO representation.

There is no fixed total-serving constraint. Nutrient adequacy controls the
amount of food, while an exact QUBO reward encourages more distinct food types.
Repeated servings receive neither a diversity reward nor a penalty.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "foods.csv"
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
NUTRIENTS = (
    "kcal", "protein_g", "carbs_g", "fat_g", "potassium_mg", "sodium_mg"
)


@dataclass(frozen=True)
class FoodData:
    foods: tuple[str, ...]
    matrix: np.ndarray
    sources: tuple[str, ...]
    serving_grams: np.ndarray


@dataclass(frozen=True)
class NutritionPolicy:
    energy_target: float
    protein_range: tuple[float, float]
    carbs_range: tuple[float, float]
    fat_range: tuple[float, float]
    potassium_minimum: float
    sodium_range: tuple[float, float]
    # energy, protein, carbs, fat, potassium, sodium-low, sodium-high
    weights: tuple[float, ...]
    diversity_reward: float
    max_servings_per_food: int


@dataclass(frozen=True)
class QuboSurrogate:
    """Quadratic binary approximation of the piecewise nutrition cost only."""

    intercept: float
    linear: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    pair_coefficients: np.ndarray

    def predict_bits(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=float)
        result = self.intercept + bits @ self.linear
        for coefficient, (i, j) in zip(self.pair_coefficients, self.pairs):
            result = result + coefficient * bits[..., i] * bits[..., j]
        return result


def load_food_data(path: Path | str = DEFAULT_DATA_PATH) -> FoodData:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No food rows found in {path}")
    return FoodData(
        foods=tuple(row["food"] for row in rows),
        matrix=np.asarray(
            [[float(row[nutrient]) for nutrient in NUTRIENTS] for row in rows],
            dtype=float,
        ),
        sources=tuple(row["source_url"] for row in rows),
        serving_grams=np.asarray([float(row["serving_g"]) for row in rows]),
    )


def load_policy(path: Path | str = DEFAULT_CONFIG_PATH) -> NutritionPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NutritionPolicy(
        energy_target=float(payload["energy_target"]),
        protein_range=tuple(map(float, payload["protein_range"])),
        carbs_range=tuple(map(float, payload["carbs_range"])),
        fat_range=tuple(map(float, payload["fat_range"])),
        potassium_minimum=float(payload["potassium_minimum"]),
        sodium_range=tuple(map(float, payload["sodium_range"])),
        weights=tuple(map(float, payload["weights"])),
        diversity_reward=float(payload["diversity_reward"]),
        max_servings_per_food=int(payload["max_servings_per_food"]),
    )


def bits_per_food(max_servings: int) -> int:
    bits = int(np.ceil(np.log2(max_servings + 1)))
    if 2**bits - 1 != max_servings:
        raise ValueError("max servings must be 2^b-1, such as 1, 3, 7, or 15")
    return bits


def all_menus(number_foods: int, max_servings: int) -> np.ndarray:
    return np.asarray(
        list(product(range(max_servings + 1), repeat=number_foods)), dtype=np.int16
    )


def menus_to_bits(menus: np.ndarray, max_servings: int) -> np.ndarray:
    menus = np.asarray(menus, dtype=np.int64)
    width = bits_per_food(max_servings)
    if np.any((menus < 0) | (menus > max_servings)):
        raise ValueError(f"Serving values must be in 0..{max_servings}")
    bits = np.empty(menus.shape[:-1] + (menus.shape[-1] * width,), dtype=np.int8)
    for bit in range(width):
        bits[..., bit::width] = (menus >> bit) & 1
    return bits


def bits_to_menus(bits: np.ndarray, number_foods: int, max_servings: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    width = bits_per_food(max_servings)
    if bits.shape[-1] != number_foods * width:
        raise ValueError("Bit-vector width does not match food data")
    menus = np.zeros(bits.shape[:-1] + (number_foods,), dtype=np.int16)
    for bit in range(width):
        menus += (2**bit) * bits[..., bit::width]
    return menus


def _target(values: np.ndarray, target: float) -> np.ndarray:
    return ((values - target) / target) ** 2


def _range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (np.maximum(lower - values, 0.0) / lower) ** 2 + (
        np.maximum(values - upper, 0.0) / upper
    ) ** 2


def _minimum(values: np.ndarray, minimum: float) -> np.ndarray:
    return (np.maximum(minimum - values, 0.0) / minimum) ** 2


def nutrition_components(
    menus: np.ndarray, data: FoodData, policy: NutritionPolicy
) -> np.ndarray:
    totals = np.asarray(menus, dtype=float) @ data.matrix
    sodium_low, sodium_high = policy.sodium_range
    return np.stack(
        [
            _target(totals[..., 0], policy.energy_target),
            _range(totals[..., 1], *policy.protein_range),
            _range(totals[..., 2], *policy.carbs_range),
            _range(totals[..., 3], *policy.fat_range),
            _minimum(totals[..., 4], policy.potassium_minimum),
            (np.maximum(sodium_low - totals[..., 5], 0.0) / sodium_low) ** 2,
            (np.maximum(totals[..., 5] - sodium_high, 0.0) / sodium_high) ** 2,
        ], axis=-1,
    )


def nutrition_cost(menus: np.ndarray, data: FoodData, policy: NutritionPolicy) -> np.ndarray:
    weights = np.asarray(policy.weights, dtype=float)
    if weights.shape != (7,):
        raise ValueError("Policy requires seven weights")
    return nutrition_components(menus, data, policy) @ weights


def distinct_food_count(menus: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.asarray(menus) > 0, axis=-1)


def diversity_value(menus: np.ndarray, policy: NutritionPolicy, number_foods: int) -> np.ndarray:
    """Negative cost: one reward per distinct food, normalized by food count."""
    return -policy.diversity_reward * distinct_food_count(menus) / number_foods


def policy_cost(menus: np.ndarray, data: FoodData, policy: NutritionPolicy) -> np.ndarray:
    return nutrition_cost(menus, data, policy) + diversity_value(
        menus, policy, len(data.foods)
    )


def quadratic_pairs(number_bits: int) -> list[tuple[int, int]]:
    return list(combinations(range(number_bits), 2))


def quadratic_features(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=float)
    pairs = quadratic_pairs(bits.shape[-1])
    result = np.empty(bits.shape[:-1] + (1 + bits.shape[-1] + len(pairs),))
    result[..., 0] = 1.0
    result[..., 1 : 1 + bits.shape[-1]] = bits
    start = 1 + bits.shape[-1]
    for column, (i, j) in enumerate(pairs, start=start):
        result[..., column] = bits[..., i] * bits[..., j]
    return result


def fit_nutrition_qubo(
    menus: np.ndarray, scores: np.ndarray, max_servings: int, ridge: float = 1e-8,
    sample_weights: np.ndarray | None = None,
) -> QuboSurrogate:
    """Fit the piecewise nutrition score; diversity is later added exactly.

    Optional weights are for *diagnostic* alternative fits. A lower local
    error does not guarantee a better QUBO optimum, so callers must validate
    the resulting ranking rather than silently select a weighted model.
    """
    bits = menus_to_bits(menus, max_servings)
    x = quadratic_features(bits)
    scores = np.asarray(scores, dtype=float)
    if sample_weights is None:
        square_root_weights = np.ones(len(menus), dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != (len(menus),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("sample_weights must be positive and match the menus")
        square_root_weights = np.sqrt(weights)
    regularizer = np.sqrt(ridge) * np.eye(x.shape[1])
    regularizer[0, 0] = 0.0
    coefficients, *_ = np.linalg.lstsq(
        np.vstack([x * square_root_weights[:, None], regularizer]),
        np.concatenate([scores * square_root_weights, np.zeros(x.shape[1])]), rcond=None,
    )
    pair_start = 1 + bits.shape[1]
    return QuboSurrogate(
        intercept=float(coefficients[0]),
        linear=coefficients[1:pair_start].copy(),
        pairs=tuple(quadratic_pairs(bits.shape[1])),
        pair_coefficients=coefficients[pair_start:].copy(),
    )


def exact_diversity_from_bits(
    bits: np.ndarray, number_foods: int, policy: NutritionPolicy
) -> np.ndarray:
    menus = bits_to_menus(bits, number_foods, policy.max_servings_per_food)
    return diversity_value(menus, policy, number_foods)


def surrogate_policy_cost_bits(
    bits: np.ndarray, surrogate: QuboSurrogate, number_foods: int, policy: NutritionPolicy
) -> np.ndarray:
    return surrogate.predict_bits(bits) + exact_diversity_from_bits(
        bits, number_foods, policy
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def validate_surrogate(
    menus: np.ndarray, true_costs: np.ndarray, surrogate: QuboSurrogate,
    policy: NutritionPolicy,
) -> dict[str, float | int | bool | None]:
    bits = menus_to_bits(menus, policy.max_servings_per_food)
    predicted = surrogate_policy_cost_bits(bits, surrogate, menus.shape[1], policy)
    true_order = np.argsort(true_costs)
    predicted_order = np.argsort(predicted)
    count = max(1, int(0.01 * len(menus)))
    surrogate_best = int(predicted_order[0])
    true_best = int(true_order[0])
    near_mask = true_costs <= true_costs[true_best] + 0.001
    near_true = true_costs[near_mask]
    near_predicted = predicted[near_mask]
    return {
        "menus": int(len(menus)),
        "rmse": float(np.sqrt(np.mean((predicted - true_costs) ** 2))),
        "pearson": correlation(true_costs, predicted),
        "spearman": correlation(rankdata(true_costs), rankdata(predicted)),
        "true_best_index": true_best,
        "surrogate_best_index": surrogate_best,
        "same_best_menu": bool(true_best == surrogate_best),
        "true_best_surrogate_rank": int(rankdata(predicted)[true_best] + 1),
        "surrogate_best_true_rank": int(rankdata(true_costs)[surrogate_best] + 1),
        "surrogate_best_true_cost": float(true_costs[surrogate_best]),
        "surrogate_best_true_cost_gap": float(true_costs[surrogate_best] - true_costs[true_best]),
        "near_optimum_width": 0.001,
        "near_optimum_menus": int(near_mask.sum()),
        "near_optimum_mae": float(np.mean(np.abs(near_predicted - near_true))),
        "near_optimum_spearman": float(correlation(rankdata(near_true), rankdata(near_predicted))) if len(near_true) > 1 else None,
        "top_1pct_overlap": float(
            len(set(true_order[:count]) & set(predicted_order[:count])) / count
        ),
    }


def build_quadratic_program(
    surrogate: QuboSurrogate, data: FoodData, policy: NutritionPolicy
):
    """Build an unconstrained QP and add the diversity reward exactly."""
    from qiskit_optimization import QuadraticProgram

    width = bits_per_food(policy.max_servings_per_food)
    if width > 2:
        raise ValueError("No-ancilla diversity QUBO supports max servings 1 or 3")
    qp = QuadraticProgram("range_aware_nutrition_with_diversity")
    names = []
    for food in data.foods:
        for bit in range(width):
            name = f"{food}@{bit}"
            qp.binary_var(name=name)
            names.append(name)

    linear = surrogate.linear.copy()
    pair_values = dict(zip(surrogate.pairs, surrogate.pair_coefficients))
    reward = policy.diversity_reward / len(data.foods)
    for food_index in range(len(data.foods)):
        food_bits = [food_index * width + bit for bit in range(width)]
        for bit_index in food_bits:
            linear[bit_index] -= reward
        if width == 2:  # -reward * OR(x0,x1)
            pair = tuple(food_bits)
            pair_values[pair] = pair_values.get(pair, 0.0) + reward

    qp.minimize(
        constant=surrogate.intercept,
        linear={name: float(value) for name, value in zip(names, linear)},
        quadratic={
            (names[i], names[j]): float(value)
            for (i, j), value in pair_values.items() if abs(value) > 1e-14
        },
    )
    return qp


def perturb_policy(
    policy: NutritionPolicy, rng: np.random.Generator, fraction: float
) -> NutritionPolicy:
    def p(value: float) -> float:
        return float(value * rng.uniform(1 - fraction, 1 + fraction))
    def ordered(bounds: tuple[float, float]) -> tuple[float, float]:
        return tuple(sorted((p(bounds[0]), p(bounds[1]))))
    return replace(
        policy,
        energy_target=p(policy.energy_target),
        protein_range=ordered(policy.protein_range),
        carbs_range=ordered(policy.carbs_range),
        fat_range=ordered(policy.fat_range),
        potassium_minimum=p(policy.potassium_minimum),
        sodium_range=ordered(policy.sodium_range),
        weights=tuple(p(value) for value in policy.weights),
    )


def format_menu(menu: np.ndarray, foods: tuple[str, ...]) -> str:
    return ", ".join(
        f"{food}x{int(value)}" for food, value in zip(foods, menu) if value > 0
    )
