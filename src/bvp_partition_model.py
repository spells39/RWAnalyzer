import os
import time
from functools import lru_cache

import numpy as np
import scipy.linalg
from scipy.optimize import fsolve
from scipy.special import expit, logit
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp


LEGACY_BVP_STRATEGY_TIE_TOL = 1e-3
DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_FLOOR = 0.65
DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_SCALE = 0.3
DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_EXPONENT = 0.75
DEFAULT_LEGACY_BVP_SOFT_SHARPEN_SCALE = 1.2
DEFAULT_LEGACY_BVP_SOFT_SHARPEN_EXPONENT = 2.0
DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_SCALE = 0.25
DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_CENTER = 0.50
DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_WIDTH = 0.25
DEFAULT_LEGACY_BVP_SCORE_SMOOTHING_WEIGHT = 0.10
DEFAULT_LEGACY_BVP_AXIS_ROLLOUT_HORIZON = 4
DEFAULT_LEGACY_BVP_AXIS_ROLLOUT_DECAY = 0.72
DEFAULT_LEGACY_BVP_AXIS_STATIC_EFFICIENCY_WEIGHT = 0.65
DEFAULT_LEGACY_BVP_AXIS_HIT_WEIGHT = 0.75
DEFAULT_LEGACY_BVP_AXIS_PROXIMITY_WEIGHT = 0.40
DEFAULT_LEGACY_BVP_SOFT_VALUE_TIEBREAK_SCALE = 0.2
DEFAULT_LEGACY_BVP_ACTIVATION_SCALE = 0.03
DEFAULT_LEGACY_BVP_ACTIVATION_EXPONENT = 1.4
DEFAULT_LEGACY_BVP_LOCAL_SCORE_NORMALIZATION_WEIGHT = 0.40
DEFAULT_LEGACY_BVP_LOCAL_SCORE_NORMALIZATION_GAIN = 0.60
DEFAULT_LEGACY_BVP_BORDER_CONTRAST_BASE = 0.90
DEFAULT_LEGACY_BVP_BORDER_CONTRAST_SCALE = 0.10
DEFAULT_LEGACY_BVP_CENTER_EPSILON_FLOOR = 0.05
DEFAULT_LEGACY_BVP_CENTER_EPSILON_MAX_WEIGHT = 0.80
DEFAULT_LEGACY_BVP_CENTER_EPSILON_EXPONENT = 2.0

OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def find_max(func, game, temperature=None):
    value_if_strategy_0 = func(0.0, game)
    value_if_strategy_1 = func(1.0, game)

    if temperature is not None:
        delta = (value_if_strategy_0 - value_if_strategy_1) / temperature
        delta = np.clip(delta, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-delta))
        value = func(p, game)
        return value, p

    if value_if_strategy_0 < value_if_strategy_1:
        return value_if_strategy_0, 0.0
    return value_if_strategy_1, 1.0


def get_win(p, game):
    return (p / 2.0) * (game[0, 0] + game[1, 0]) + ((1.0 - p) / 2.0) * (game[0, 1] + game[1, 1])


def get_value(game):
    return find_max(get_win, game)


def compute_legacy_bvp_soft_temperature(epsilon):
    if epsilon <= 0.0:
        return None
    return (
        DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_FLOOR
        + DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_SCALE * (epsilon ** DEFAULT_LEGACY_BVP_SOFT_TEMPERATURE_EXPONENT)
    )


def sharpen_legacy_bvp_probability(p, epsilon):
    if epsilon <= 0.0:
        return p
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    mid_boost = DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_SCALE * np.exp(
        -(
            (epsilon - DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_CENTER)
            / DEFAULT_LEGACY_BVP_SOFT_SHARPEN_MID_BOOST_WIDTH
        )
        ** 2
    )
    sharpen = 1.0 + DEFAULT_LEGACY_BVP_SOFT_SHARPEN_SCALE * (
        epsilon ** DEFAULT_LEGACY_BVP_SOFT_SHARPEN_EXPONENT
    ) + mid_boost
    return float(expit(logit(p) * sharpen))


def compute_legacy_bvp_baseline_blend_weight(epsilon):
    if epsilon <= 0.0:
        return 0.0
    scaled_epsilon = epsilon / DEFAULT_LEGACY_BVP_ACTIVATION_SCALE
    return float(1.0 - np.exp(-(scaled_epsilon ** DEFAULT_LEGACY_BVP_ACTIVATION_EXPONENT)))


def get_value_with_border_tiebreak(game, global_index, global_n, epsilon, geometry_context=None):
    value_if_strategy_0 = get_win(0.0, game)
    value_if_strategy_1 = get_win(1.0, game)

    if geometry_context is not None:
        quarter_preference = geometry_context["quarter_preference"][global_index]
        baseline_p = geometry_context["baseline_probability"][global_index]
        score_delta = geometry_context["smoothed_score_delta"][global_index]
        border_contrast = geometry_context.get("border_contrast", {}).get(global_index, 1.0)
    else:
        quarter_preference = _legacy_quarter_border_tiebreak(global_index, global_n)
        baseline_p = _legacy_quarter_border_baseline_probability(global_index, global_n)
        raw_score_delta = _legacy_bvp_smoothed_score_delta(global_index, global_n)
        score_delta = _legacy_bvp_locally_normalized_score_delta(global_index, global_n, raw_score_delta)
        border_contrast = _legacy_bvp_border_contrast(global_index, global_n)

    if epsilon <= 0.0:
        if quarter_preference == "vertical":
            return value_if_strategy_1, 1.0
        if quarter_preference == "horizontal":
            return value_if_strategy_0, 0.0

    if abs(value_if_strategy_1 - value_if_strategy_0) <= LEGACY_BVP_STRATEGY_TIE_TOL:
        if quarter_preference == "vertical":
            return value_if_strategy_1, 1.0
        if quarter_preference == "horizontal":
            return value_if_strategy_0, 0.0

    if epsilon <= 0.0:
        return find_max(get_win, game, temperature=None)

    score_strength = compute_legacy_bvp_soft_temperature(epsilon)
    value_margin = value_if_strategy_1 - value_if_strategy_0
    value_tiebreak = np.tanh(value_margin / max(LEGACY_BVP_STRATEGY_TIE_TOL, 1e-9))
    score_logit = border_contrast * (
        score_strength * score_delta + DEFAULT_LEGACY_BVP_SOFT_VALUE_TIEBREAK_SCALE * epsilon * value_tiebreak
    )
    score_p = float(expit(score_logit))
    blend_weight = compute_legacy_bvp_baseline_blend_weight(epsilon)
    p = (1.0 - blend_weight) * baseline_p + blend_weight * score_p
    p = sharpen_legacy_bvp_probability(p, epsilon)
    return get_win(p, game), p


def inner_n_to_global_N(index, inner_n, global_n):
    row = index // inner_n
    col = index % inner_n
    return (row + 1) * global_n + (col + 1)


def global_N_to_inner_n(index, inner_n, global_n):
    row = index // global_n
    col = index % global_n

    if row < 1 or col < 1 or row > global_n - 2 or col > global_n - 2:
        raise ValueError("Index should match an inner node.")
    return (row - 1) * inner_n + (col - 1)


def get_row_col(global_index, global_n):
    return global_index // global_n, global_index % global_n


def move_global_index(global_index, global_n, direction):
    row, col = get_row_col(global_index, global_n)
    dr, dc = OFFSETS[direction]
    return (row + dr) * global_n + (col + dc)


def move_global_index_absorbing(global_index, global_n, direction):
    row, col = get_row_col(global_index, global_n)
    if row == 0 or row == global_n - 1 or col == 0 or col == global_n - 1:
        return global_index
    dr, dc = OFFSETS[direction]
    next_row = min(max(row + dr, 0), global_n - 1)
    next_col = min(max(col + dc, 0), global_n - 1)
    return next_row * global_n + next_col


@lru_cache(maxsize=None)
def compute_distance_to_center(global_index, global_n):
    center_row = global_n // 2
    center_col = global_n // 2
    row, col = get_row_col(global_index, global_n)
    return abs(row - center_row) + abs(col - center_col)


@lru_cache(maxsize=None)
def compute_distance_to_border(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    distances = [row, global_n - 1 - row, col, global_n - 1 - col]
    return min(distances)


def compute_radius(global_n):
    return global_n // 4


def _legacy_preferred_border_strategy(global_index, global_n):
    vertical_best = min(
        compute_distance_to_border(move_global_index(global_index, global_n, "up"), global_n),
        compute_distance_to_border(move_global_index(global_index, global_n, "down"), global_n),
    )
    horizontal_best = min(
        compute_distance_to_border(move_global_index(global_index, global_n, "left"), global_n),
        compute_distance_to_border(move_global_index(global_index, global_n, "right"), global_n),
    )

    if vertical_best < horizontal_best:
        return "vertical"
    if horizontal_best < vertical_best:
        return "horizontal"
    return None


@lru_cache(maxsize=None)
def _legacy_quarter_border_tiebreak(global_index, global_n):
    center_row = global_n // 2
    center_col = global_n // 2
    row, col = get_row_col(global_index, global_n)

    vertical_offset = abs(row - center_row)
    horizontal_offset = abs(col - center_col)

    if vertical_offset > horizontal_offset:
        return "vertical"
    if horizontal_offset > vertical_offset:
        return "horizontal"
    return None


@lru_cache(maxsize=None)
def _legacy_quarter_border_baseline_probability(global_index, global_n):
    preferred_strategy = _legacy_quarter_border_tiebreak(global_index, global_n)
    if preferred_strategy == "vertical":
        return 1.0
    if preferred_strategy == "horizontal":
        return 0.0
    return 0.5


@lru_cache(maxsize=None)
def _legacy_axis_edge_distances(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    return (
        row,
        global_n - 1 - row,
        col,
        global_n - 1 - col,
    )


@lru_cache(maxsize=None)
def _legacy_axis_border_distance(global_index, global_n, axis):
    d_top, d_bottom, d_left, d_right = _legacy_axis_edge_distances(global_index, global_n)
    if axis == "vertical":
        return min(d_top, d_bottom)
    return min(d_left, d_right)


@lru_cache(maxsize=None)
def _legacy_axis_escape_efficiency(global_index, global_n, axis):
    d_top, d_bottom, d_left, d_right = _legacy_axis_edge_distances(global_index, global_n)
    if axis == "vertical":
        return 1.0 / ((d_top + 1.0) * (d_bottom + 1.0))
    return 1.0 / ((d_left + 1.0) * (d_right + 1.0))


@lru_cache(maxsize=None)
def _legacy_axis_hits_border(global_index, global_n, axis):
    row, col = get_row_col(global_index, global_n)
    if axis == "vertical":
        return row == 0 or row == global_n - 1
    return col == 0 or col == global_n - 1


@lru_cache(maxsize=None)
def _legacy_axis_rollout_distribution(global_index, global_n, axis, steps):
    if steps == 0:
        return ((global_index, 1.0),)

    aggregated = {}
    directions = ("up", "down") if axis == "vertical" else ("left", "right")
    for state, probability in _legacy_axis_rollout_distribution(global_index, global_n, axis, steps - 1):
        if _legacy_axis_hits_border(state, global_n, axis):
            aggregated[state] = aggregated.get(state, 0.0) + probability
            continue

        for direction in directions:
            next_state = move_global_index_absorbing(state, global_n, direction)
            aggregated[next_state] = aggregated.get(next_state, 0.0) + probability * 0.5

    return tuple(sorted(aggregated.items()))


@lru_cache(maxsize=None)
def _legacy_axis_escape_score(global_index, global_n, axis):
    static_efficiency = _legacy_axis_escape_efficiency(global_index, global_n, axis)
    score = 0.0

    for steps in range(1, DEFAULT_LEGACY_BVP_AXIS_ROLLOUT_HORIZON + 1):
        rollout = _legacy_axis_rollout_distribution(global_index, global_n, axis, steps)
        hit_probability = 0.0
        expected_proximity = 0.0

        for state, probability in rollout:
            axis_distance = _legacy_axis_border_distance(state, global_n, axis)
            if _legacy_axis_hits_border(state, global_n, axis):
                hit_probability += probability

            proximity = 1.0 / (axis_distance + 1.0)
            expected_proximity += probability * proximity

        weight = DEFAULT_LEGACY_BVP_AXIS_ROLLOUT_DECAY ** (steps - 1)
        score += weight * (
            DEFAULT_LEGACY_BVP_AXIS_HIT_WEIGHT * hit_probability
            + DEFAULT_LEGACY_BVP_AXIS_PROXIMITY_WEIGHT * expected_proximity
        )

    return DEFAULT_LEGACY_BVP_AXIS_STATIC_EFFICIENCY_WEIGHT * static_efficiency + score


@lru_cache(maxsize=None)
def _legacy_bvp_score_delta(global_index, global_n):
    vertical_score = _legacy_axis_escape_score(global_index, global_n, "vertical")
    horizontal_score = _legacy_axis_escape_score(global_index, global_n, "horizontal")
    normalization = abs(vertical_score) + abs(horizontal_score) + 1e-12
    delta = (vertical_score - horizontal_score) / normalization

    if abs(delta) < 1e-9:
        quarter_preference = _legacy_quarter_border_tiebreak(global_index, global_n)
        if quarter_preference == "vertical":
            return 1e-6
        if quarter_preference == "horizontal":
            return -1e-6
    return delta


@lru_cache(maxsize=None)
def _legacy_bvp_smoothed_score_delta(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    deltas = [_legacy_bvp_score_delta(global_index, global_n)]

    for dr, dc in OFFSETS.values():
        next_row = row + dr
        next_col = col + dc
        if 1 <= next_row <= global_n - 2 and 1 <= next_col <= global_n - 2:
            neighbor_index = next_row * global_n + next_col
            deltas.append(_legacy_bvp_score_delta(neighbor_index, global_n))

    if len(deltas) == 1:
        return deltas[0]

    self_delta = deltas[0]
    neighbor_mean = float(np.mean(deltas[1:]))
    return (1.0 - DEFAULT_LEGACY_BVP_SCORE_SMOOTHING_WEIGHT) * self_delta + DEFAULT_LEGACY_BVP_SCORE_SMOOTHING_WEIGHT * neighbor_mean


def _legacy_bvp_locally_normalized_score_delta(global_index, global_n, raw_score_delta):
    weight = DEFAULT_LEGACY_BVP_LOCAL_SCORE_NORMALIZATION_WEIGHT
    if weight <= 0.0:
        return raw_score_delta

    row, col = get_row_col(global_index, global_n)
    local_values = [raw_score_delta]

    for dr, dc in OFFSETS.values():
        next_row = row + dr
        next_col = col + dc
        if 1 <= next_row <= global_n - 2 and 1 <= next_col <= global_n - 2:
            neighbor_index = next_row * global_n + next_col
            local_values.append(_legacy_bvp_smoothed_score_delta(neighbor_index, global_n))

    local_scale = float(np.mean(np.abs(local_values))) + 1e-9
    local_score_delta = float(np.tanh(DEFAULT_LEGACY_BVP_LOCAL_SCORE_NORMALIZATION_GAIN * raw_score_delta / local_scale))
    return (1.0 - weight) * raw_score_delta + weight * local_score_delta


@lru_cache(maxsize=None)
def _legacy_bvp_border_contrast(global_index, global_n):
    max_border_distance = max(1.0, (global_n - 1) / 2.0)
    border_distance = compute_distance_to_border(global_index, global_n)
    border_closeness = 1.0 - ((border_distance - 1.0) / max(1.0, max_border_distance - 1.0))
    border_closeness = float(np.clip(border_closeness, 0.0, 1.0))
    return DEFAULT_LEGACY_BVP_BORDER_CONTRAST_BASE + DEFAULT_LEGACY_BVP_BORDER_CONTRAST_SCALE * border_closeness


def _legacy_bvp_center_epsilon_weight(center_distance, radius):
    if radius <= 0 or center_distance > radius:
        return 1.0
    if center_distance <= 0:
        return 0.0

    normalized_distance = float(center_distance) / float(radius)
    return DEFAULT_LEGACY_BVP_CENTER_EPSILON_FLOOR + (
        DEFAULT_LEGACY_BVP_CENTER_EPSILON_MAX_WEIGHT - DEFAULT_LEGACY_BVP_CENTER_EPSILON_FLOOR
    ) * (normalized_distance ** DEFAULT_LEGACY_BVP_CENTER_EPSILON_EXPONENT)


def build_legacy_bvp_geometry_context(n):
    inner_n = n - 2
    baseline_probability = {}
    smoothed_score_delta = {}
    border_contrast = {}
    quarter_preference = {}
    center_distance = {}
    geometric_preferred_strategy = {}

    for i in range(inner_n ** 2):
        global_index = inner_n_to_global_N(i, inner_n, n)
        baseline_probability[global_index] = _legacy_quarter_border_baseline_probability(global_index, n)
        raw_score_delta = _legacy_bvp_smoothed_score_delta(global_index, n)
        smoothed_score_delta[global_index] = _legacy_bvp_locally_normalized_score_delta(global_index, n, raw_score_delta)
        border_contrast[global_index] = _legacy_bvp_border_contrast(global_index, n)
        quarter_preference[global_index] = _legacy_quarter_border_tiebreak(global_index, n)
        center_distance[global_index] = compute_distance_to_center(global_index, n)
        geometric_preferred_strategy[global_index] = _legacy_preferred_border_strategy(global_index, n)

    return {
        "baseline_probability": baseline_probability,
        "smoothed_score_delta": smoothed_score_delta,
        "border_contrast": border_contrast,
        "quarter_preference": quarter_preference,
        "center_distance": center_distance,
        "geometric_preferred_strategy": geometric_preferred_strategy,
    }


def _legacy_base_transition_value(next_global_index, w, n, inner_n, border_cases):
    if next_global_index in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(next_global_index, inner_n, n)
    return w[next_inner_index] + 1.0


def _legacy_preferred_border_strategy_by_value(index, w, n, inner_n, border_cases):
    vertical_value = 0.5 * (
        _legacy_base_transition_value(index - n, w, n, inner_n, border_cases)
        + _legacy_base_transition_value(index + n, w, n, inner_n, border_cases)
    )
    horizontal_value = 0.5 * (
        _legacy_base_transition_value(index + 1, w, n, inner_n, border_cases)
        + _legacy_base_transition_value(index - 1, w, n, inner_n, border_cases)
    )

    if vertical_value < horizontal_value:
        return "vertical"
    if horizontal_value < vertical_value:
        return "horizontal"
    return None


def ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices):
    os.makedirs(output_absorption_images1, exist_ok=True)
    os.makedirs(output_absorption_images2, exist_ok=True)
    os.makedirs(output_absorption_images3, exist_ok=True)
    os.makedirs(qr_matrices, exist_ok=True)


def make_banded_matrix(A, N):
    banded_matrix = np.zeros((2 * N + 1, N ** 2), dtype=np.float64)
    for i in [-N, -1, 0, 1, N]:
        d = np.diagonal(A, -i)
        if i < 0:
            banded_matrix[i + N, -i:] = d
        elif i > 0:
            banded_matrix[i + N, :-i] = d
        else:
            banded_matrix[i + N, :] = d
    return banded_matrix


def find_mean_time_banded(A, N):
    banded_matrix = make_banded_matrix(np.eye(A.shape[0]) - A, N)
    mean_times = scipy.linalg.solve_banded((N, N), banded_matrix, np.ones(banded_matrix.shape[1]))
    return mean_times[N * (N // 2) + N // 2], mean_times


# Legacy epsilon_border-style BvP solver is preserved below as *_legacy helpers.


def _legacy_compute_epsilon_border(index, w, epsilon, direction, radius, global_n, inner_n, border_cases, geometry_context=None):
    global_index = index
    if geometry_context is not None:
        cur_distance = geometry_context["center_distance"][global_index]
    else:
        cur_distance = compute_distance_to_center(global_index, global_n)

    center_epsilon_weight = _legacy_bvp_center_epsilon_weight(cur_distance, radius)

    preferred_strategy = _legacy_preferred_border_strategy_by_value(index, w, global_n, inner_n, border_cases)
    if preferred_strategy is None:
        if geometry_context is not None:
            preferred_strategy = geometry_context["geometric_preferred_strategy"][global_index]
        else:
            preferred_strategy = _legacy_preferred_border_strategy(global_index, global_n)
    if preferred_strategy is None:
        return 0.0

    direction_strategy = "vertical" if direction in ("up", "down") else "horizontal"
    if direction_strategy == preferred_strategy:
        return center_epsilon_weight * epsilon
    return -center_epsilon_weight * epsilon


def _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context=None):
    game = np.zeros((2, 2))
    game[0, 0] = _legacy_compute_a11(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
    game[0, 1] = _legacy_compute_a12(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
    game[1, 0] = _legacy_compute_a21(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
    game[1, 1] = _legacy_compute_a22(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
    return game


def _legacy_compute_a11(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context=None):
    if (index - n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - n, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(index, w, epsilon, "up", radius, n, inner_n, border_cases, geometry_context)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a21(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context=None):
    if (index + n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + n, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(index, w, epsilon, "down", radius, n, inner_n, border_cases, geometry_context)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a12(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context=None):
    if (index + 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(index, w, epsilon, "right", radius, n, inner_n, border_cases, geometry_context)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a22(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context=None):
    if (index - 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(index, w, epsilon, "left", radius, n, inner_n, border_cases, geometry_context)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_prepare_equations(w, epsilon, n, inner_n, radius, border_cases, geometry_context=None):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
        v, _ = get_value_with_border_tiebreak(game_mx, index, n, epsilon, geometry_context)
        eqs[i] = w[i] - v
    return tuple(eqs)


def _legacy_compute_state_values(w, epsilon, n, inner_n, radius, border_cases, geometry_context=None):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases, geometry_context)
        v, q1 = get_value_with_border_tiebreak(game_mx, index, n, epsilon, geometry_context)
        p1s.append(0.5)
        q1s.append(q1)
        vs.append(v)
    return np.array(p1s), np.array(q1s), np.array(vs)


def solve_bvp_sweep_legacy(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    max_attempts=100,
):
    N = n - 1
    inner_n = n - 2
    radius = compute_radius(n)
    border_cases = get_border_cases(N)
    geometry_context = build_legacy_bvp_geometry_context(n)

    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)
    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    w_new_list = []
    strategy_snapshots = []
    mean_times = []
    solve_times = []
    attempt_counts = []
    previous_solution = None
    previous_previous_solution = None

    for epsilon in tqdm(epsilon_values, desc="Solving legacy equations"):
        epsilon_start = time.perf_counter()
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            if previous_solution is not None and previous_previous_solution is not None and attempts == 1:
                starting_params = previous_solution + (previous_solution - previous_previous_solution)
            elif previous_solution is not None and attempts == 1:
                starting_params = previous_solution
            elif previous_solution is not None:
                starting_params = previous_solution + 0.05 * np.random.standard_normal(inner_n ** 2)
            else:
                starting_params = np.random.random(inner_n ** 2) * (inner_n - 1) ** 2
            w_new, _, _, message = fsolve(
                lambda w: _legacy_prepare_equations(w, epsilon, n, inner_n, radius, border_cases, geometry_context),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        previous_previous_solution = previous_solution
        previous_solution = w_new
        attempt_counts.append(attempts)
        solve_times.append(time.perf_counter() - epsilon_start)
        p1_flat, q1_flat, v_flat = _legacy_compute_state_values(w_new, epsilon, n, inner_n, radius, border_cases, geometry_context)
        p1_matrix = np.reshape(p1_flat, (inner_n, inner_n))
        q1_matrix = np.reshape(q1_flat, (inner_n, inner_n))

        qr_optimal, probability_optimal = make_prob_matrix(
            N,
            np.pad(p1_matrix, pad_width=1, mode="constant", constant_values=0).T,
            np.pad(q1_matrix, pad_width=1, mode="constant", constant_values=0).T,
        )
        mean_time, _ = find_mean_time_banded(probability_optimal, N - 1)
        mean_times.append(mean_time)
        np.save(qr_matrices + f"qr_{epsilon:.2f}", qr_optimal)

        strategy_snapshots.append(
            {
                "epsilon": float(epsilon),
                "p1s": p1_flat,
                "q1s": q1_flat,
                "vs": v_flat,
                "w": w_new,
                "mean_time": float(mean_time),
                "solve_time_sec": float(solve_times[-1]),
                "attempts": int(attempt_counts[-1]),
            }
        )

    with open(output_duration + "duration.txt", "w") as file:
        for mean_time in mean_times:
            file.write(f"{mean_time} ")

    with open(output_duration + "solve_times.txt", "w") as file:
        for solve_time in solve_times:
            file.write(f"{solve_time} ")

    with open(output_duration + "attempt_counts.txt", "w") as file:
        for attempts in attempt_counts:
            file.write(f"{attempts} ")

    return {
        "n": n,
        "N": N,
        "inner_n": inner_n,
        "radius": radius,
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": w_new_list,
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "solve_times": np.array(solve_times, dtype=float),
        "attempt_counts": np.array(attempt_counts, dtype=int),
        "mode": "border-legacy",
    }
