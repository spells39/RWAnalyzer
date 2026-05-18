import os

import numpy as np
import scipy.linalg
from scipy.spatial.distance import jensenshannon
from scipy.special import expit
from scipy.optimize import fsolve
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp

LEGACY_PVB_VALUE_TIE_TOL = 1e-6
LEGACY_PVB_STRATEGY_TIE_TOL = 1e-3
DEFAULT_PVB_BOUNDED_BIAS_SCALE = 0.46
DEFAULT_PVB_BOUNDED_BIAS_TAU = 0.30
DEFAULT_PVB_BOUNDED_BIAS_POWER = 0.75
DEFAULT_PVB_SOFT_RESPONSE_MIN_TEMPERATURE = 0.15
DEFAULT_PVB_SOFT_RESPONSE_TEMPERATURE_SCALE = 0.70
DEFAULT_PVB_SOFT_RESPONSE_TEMPERATURE_TAU = 0.50
DEFAULT_PVB_AMBIGUITY_MIX_SCALE = 0.72
DEFAULT_PVB_AMBIGUITY_MIX_RISE = 0.28
DEFAULT_PVB_AMBIGUITY_SCALE = 0.30
DEFAULT_PVB_GEOMETRY_MIX_SCALE = 0.10
DEFAULT_PVB_GEOMETRY_MIX_RISE = 0.28
DEFAULT_PVB_GEOMETRY_SCORE_SCALE = 0.75


def compute_pvb_bounded_bias(epsilon):
    if epsilon <= 0.0:
        return 0.0
    normalized = (float(epsilon) / DEFAULT_PVB_BOUNDED_BIAS_TAU) ** DEFAULT_PVB_BOUNDED_BIAS_POWER
    return float(DEFAULT_PVB_BOUNDED_BIAS_SCALE * np.tanh(normalized))


def compute_pvb_soft_response_temperature(epsilon):
    if epsilon <= 0.0:
        return None
    return float(
        DEFAULT_PVB_SOFT_RESPONSE_MIN_TEMPERATURE
        + DEFAULT_PVB_SOFT_RESPONSE_TEMPERATURE_SCALE
        * np.exp(-float(epsilon) / DEFAULT_PVB_SOFT_RESPONSE_TEMPERATURE_TAU)
    )


def compute_pvb_geometry_mix_weight(epsilon):
    if epsilon <= 0.0:
        return 0.0
    return float(
        DEFAULT_PVB_GEOMETRY_MIX_SCALE
        * (1.0 - np.exp(-float(epsilon) / DEFAULT_PVB_GEOMETRY_MIX_RISE))
    )


def compute_pvb_ambiguity_mix(epsilon, value_delta):
    if epsilon <= 0.0:
        return np.zeros_like(value_delta, dtype=float)
    epsilon_weight = DEFAULT_PVB_AMBIGUITY_MIX_SCALE * (
        1.0 - np.exp(-float(epsilon) / DEFAULT_PVB_AMBIGUITY_MIX_RISE)
    )
    ambiguity = np.exp(-np.abs(value_delta) / DEFAULT_PVB_AMBIGUITY_SCALE)
    return epsilon_weight * ambiguity


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


def compute_distance_to_center(global_index, global_n):
    center_row = global_n // 2
    center_col = global_n // 2
    row, col = get_row_col(global_index, global_n)
    return abs(row - center_row) + abs(col - center_col)


def compute_distance_to_border(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    distances = [row, global_n - 1 - row, col, global_n - 1 - col]
    return min(distances)


def compute_radius(global_n):
    return global_n // 4


def move_global_index(global_index, global_n, direction):
    offsets = {
        "up": -global_n,
        "down": global_n,
        "left": -1,
        "right": 1,
    }
    return global_index + offsets[direction]


def _legacy_base_transition_value(next_global_index, w, n, inner_n, border_cases):
    if next_global_index in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(next_global_index, inner_n, n)
    return w[next_inner_index] + 1.0


def _legacy_preferred_center_strategy(global_index, global_n):
    up_right_distance = 0.5 * (
        compute_distance_to_center(move_global_index(global_index, global_n, "up"), global_n)
        + compute_distance_to_center(move_global_index(global_index, global_n, "right"), global_n)
    )
    down_left_distance = 0.5 * (
        compute_distance_to_center(move_global_index(global_index, global_n, "down"), global_n)
        + compute_distance_to_center(move_global_index(global_index, global_n, "left"), global_n)
    )

    if up_right_distance < down_left_distance:
        return "up_right"
    if down_left_distance < up_right_distance:
        return "down_left"
    return None


def _legacy_preferred_center_strategy_by_value(index, w, n, inner_n, border_cases):
    up_right_value = 0.5 * (
        _legacy_base_transition_value(index - n, w, n, inner_n, border_cases)
        + _legacy_base_transition_value(index + 1, w, n, inner_n, border_cases)
    )
    down_left_value = 0.5 * (
        _legacy_base_transition_value(index + n, w, n, inner_n, border_cases)
        + _legacy_base_transition_value(index - 1, w, n, inner_n, border_cases)
    )

    if up_right_value > down_left_value:
        return "up_right"
    if down_left_value > up_right_value:
        return "down_left"
    return None


def _legacy_diagonal_center_tiebreak(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    if row > col:
        return "up_right"
    if row < col:
        return "down_left"
    return None


def _legacy_preferred_center_strategy_with_baseline(index, w, global_n, inner_n, border_cases):
    global_index = index
    baseline_strategy = _legacy_diagonal_center_tiebreak(global_index, global_n)
    if baseline_strategy is not None:
        return baseline_strategy

    preferred_strategy = None
    if w is not None and border_cases is not None:
        preferred_strategy = _legacy_preferred_center_strategy_by_value(index, w, global_n, inner_n, border_cases)
    if preferred_strategy is not None:
        return preferred_strategy

    return _legacy_preferred_center_strategy(global_index, global_n)


def compute_epsilon_border(index, w, epsilon, direction, radius, global_n, inner_n, border_cases):
    global_index = index
    cur_distance = compute_distance_to_center(global_index, global_n)

    if cur_distance <= radius:
        return 0.0

    preferred_strategy = _legacy_preferred_center_strategy_with_baseline(
        index,
        w,
        global_n,
        inner_n,
        border_cases,
    )
    if preferred_strategy is None:
        return 0.0

    direction_strategy = "up_right" if direction in ("up", "right") else "down_left"
    bounded_bias = compute_pvb_bounded_bias(epsilon)
    if direction_strategy == preferred_strategy:
        return bounded_bias
    return -bounded_bias


def _strategy_to_sign(strategy):
    if strategy == "up_right":
        return 1.0
    if strategy == "down_left":
        return -1.0
    return 0.0


def build_pvb_solver_cache(n, inner_n, radius):
    state_count = inner_n ** 2
    global_indices = np.empty(state_count, dtype=np.int64)
    next_inner_indices = np.full((state_count, 4), -1, dtype=np.int64)
    border_moves = np.zeros((state_count, 4), dtype=bool)
    diagonal_preferred_sign = np.zeros(state_count, dtype=float)
    geometry_preferred_sign = np.zeros(state_count, dtype=float)
    center_suppressed = np.zeros(state_count, dtype=bool)
    geometric_center_score_delta = np.zeros(state_count, dtype=float)

    # Direction order: up, right, down, left. This matches the local game layout.
    offsets = np.array([-n, 1, n, -1], dtype=np.int64)
    for inner_index in range(state_count):
        global_index = inner_n_to_global_N(inner_index, inner_n, n)
        global_indices[inner_index] = global_index
        center_suppressed[inner_index] = compute_distance_to_center(global_index, n) <= radius
        diagonal_preferred_sign[inner_index] = _strategy_to_sign(
            _legacy_diagonal_center_tiebreak(global_index, n)
        )
        geometry_preferred_sign[inner_index] = _strategy_to_sign(
            _legacy_preferred_center_strategy(global_index, n)
        )
        geometric_center_score_delta[inner_index] = _legacy_geometric_center_score_delta(
            global_index,
            n,
        )

        for direction_index, offset in enumerate(offsets):
            next_global_index = global_index + int(offset)
            next_row, next_col = get_row_col(next_global_index, n)
            if 1 <= next_row <= n - 2 and 1 <= next_col <= n - 2:
                next_inner_indices[inner_index, direction_index] = global_N_to_inner_n(
                    next_global_index,
                    inner_n,
                    n,
                )
            else:
                border_moves[inner_index, direction_index] = True

    return {
        "global_indices": global_indices,
        "next_inner_indices": next_inner_indices,
        "border_moves": border_moves,
        "diagonal_preferred_sign": diagonal_preferred_sign,
        "geometry_preferred_sign": geometry_preferred_sign,
        "center_suppressed": center_suppressed,
        "geometric_center_score_delta": geometric_center_score_delta,
        "direction_signs": np.array([1.0, 1.0, -1.0, -1.0], dtype=float),
    }


def compute_cached_base_values(w, solver_cache):
    base_values = np.ones_like(solver_cache["next_inner_indices"], dtype=float)
    next_inner_indices = solver_cache["next_inner_indices"]
    border_moves = solver_cache["border_moves"]
    for direction_index in range(4):
        inside_mask = ~border_moves[:, direction_index]
        base_values[inside_mask, direction_index] = (
            w[next_inner_indices[inside_mask, direction_index]] + 1.0
        )
    return base_values


def compute_cached_preferred_signs(base_values, solver_cache):
    up_right_value = 0.5 * (base_values[:, 0] + base_values[:, 1])
    down_left_value = 0.5 * (base_values[:, 2] + base_values[:, 3])

    value_preferred_sign = np.zeros(base_values.shape[0], dtype=float)
    value_preferred_sign[up_right_value > down_left_value] = 1.0
    value_preferred_sign[down_left_value > up_right_value] = -1.0

    preferred_signs = solver_cache["diagonal_preferred_sign"].copy()
    tied_mask = preferred_signs == 0.0
    preferred_signs[tied_mask] = value_preferred_sign[tied_mask]

    tied_mask = preferred_signs == 0.0
    preferred_signs[tied_mask] = solver_cache["geometry_preferred_sign"][tied_mask]

    preferred_signs[solver_cache["center_suppressed"]] = 0.0
    return preferred_signs


def compute_cached_game_values(w, epsilon, solver_cache):
    base_values = compute_cached_base_values(w, solver_cache)
    preferred_signs = compute_cached_preferred_signs(base_values, solver_cache)
    bounded_bias = compute_pvb_bounded_bias(epsilon)
    epsilon_adjustments = (
        bounded_bias
        * preferred_signs[:, None]
        * solver_cache["direction_signs"][None, :]
    )
    epsilon_adjustments = np.where(solver_cache["border_moves"], 0.0, epsilon_adjustments)
    return base_values + epsilon_adjustments


def _legacy_geometric_center_score_delta(global_index, global_n):
    strategy_zero_score = (
        compute_directional_safety_score(global_index, "up", global_n)
        + compute_directional_safety_score(global_index, "right", global_n)
    ) / 2.0
    strategy_one_score = (
        compute_directional_safety_score(global_index, "down", global_n)
        + compute_directional_safety_score(global_index, "left", global_n)
    ) / 2.0
    return strategy_one_score - strategy_zero_score


def compute_cached_values_and_probabilities(game_values, epsilon, global_n, solver_cache, player_temperature=None):
    value_if_strategy_1 = 0.5 * (game_values[:, 0] + game_values[:, 1])
    value_if_strategy_0 = 0.5 * (game_values[:, 2] + game_values[:, 3])

    if epsilon <= 0.0 and player_temperature is None:
        p1s = (value_if_strategy_1 > value_if_strategy_0).astype(float)
        tied_mask = np.abs(value_if_strategy_1 - value_if_strategy_0) <= LEGACY_PVB_STRATEGY_TIE_TOL
        diagonal_sign = solver_cache["diagonal_preferred_sign"]
        p1s[tied_mask & (diagonal_sign > 0.0)] = 1.0
        p1s[tied_mask & (diagonal_sign < 0.0)] = 0.0
        p1s[tied_mask & (diagonal_sign == 0.0)] = 0.5
    else:
        effective_temperature = player_temperature
        if effective_temperature is None:
            effective_temperature = compute_pvb_soft_response_temperature(epsilon)

        delta = (value_if_strategy_1 - value_if_strategy_0) / effective_temperature
        delta = np.clip(delta, -60.0, 60.0)
        p1s = 1.0 / (1.0 + np.exp(-delta))

        geometry_p = 1.0 - expit(
            DEFAULT_PVB_GEOMETRY_SCORE_SCALE
            * epsilon
            * solver_cache["geometric_center_score_delta"]
        )
        geometry_weight = compute_pvb_geometry_mix_weight(epsilon)
        p1s = (1.0 - geometry_weight) * p1s + geometry_weight * geometry_p

        value_delta = value_if_strategy_1 - value_if_strategy_0
        ambiguity_mix = compute_pvb_ambiguity_mix(epsilon, value_delta)
        p1s = (1.0 - ambiguity_mix) * p1s + ambiguity_mix * 0.5

    values = p1s * value_if_strategy_1 + (1.0 - p1s) * value_if_strategy_0
    return values, p1s


def prepare_equations_cached(w, epsilon, n, inner_n, radius, border_cases, solver_cache, player_temperature=None):
    del n, inner_n, radius, border_cases

    game_values = compute_cached_game_values(w, epsilon, solver_cache)
    values, _ = compute_cached_values_and_probabilities(
        game_values,
        epsilon,
        None,
        solver_cache,
        player_temperature=player_temperature,
    )
    return tuple(w - values)


def compute_state_values_cached(w, epsilon, n, inner_n, radius, border_cases, solver_cache, player_temperature=None):
    del n, inner_n, radius, border_cases

    game_values = compute_cached_game_values(w, epsilon, solver_cache)
    vs, p1s = compute_cached_values_and_probabilities(
        game_values,
        epsilon,
        None,
        solver_cache,
        player_temperature=player_temperature,
    )
    q1s = np.full(len(w), 0.5)
    return p1s, q1s, vs


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


def validate_pvb_geometry(n, inner_n, radius):
    for inner_index in range(inner_n ** 2):
        global_index = inner_n_to_global_N(inner_index, inner_n, n)
        row, col = get_row_col(global_index, n)
        current_center_distance = compute_distance_to_center(global_index, n)
        preferred_strategy = _legacy_preferred_center_strategy_with_baseline(
            global_index,
            None,
            n,
            inner_n,
            None,
        )

        for direction, (dr, dc) in {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }.items():
            next_row = row + dr
            next_col = col + dc
            if not (1 <= next_row <= n - 2 and 1 <= next_col <= n - 2):
                continue

            epsilon_value = compute_epsilon_border(global_index, None, 1.0, direction, radius, n, inner_n, None)
            epsilon_sign = np.sign(epsilon_value)
            if preferred_strategy is None:
                expected_sign = 0.0
            else:
                use_up_right = direction in ("up", "right")
                expected_sign = 1.0 if (use_up_right == (preferred_strategy == "up_right")) else -1.0

            if current_center_distance <= radius and epsilon_sign != 0.0:
                raise AssertionError("Center radius should suppress epsilon.")

            if current_center_distance > radius:
                if epsilon_sign != expected_sign:
                    raise AssertionError(
                        f"Unexpected epsilon sign for state={inner_index}, direction={direction}."
                    )


def build_symmetric_legacy_start(inner_n, global_n):
    values = np.zeros(inner_n ** 2, dtype=float)
    for inner_index in range(inner_n ** 2):
        global_index = inner_n_to_global_N(inner_index, inner_n, global_n)
        # Symmetric proxy for absorption time: farther from border => larger value.
        values[inner_index] = float(compute_distance_to_border(global_index, global_n) + 1.0)
    return values


def ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices):
    os.makedirs(output_absorption_images1, exist_ok=True)
    os.makedirs(output_absorption_images2, exist_ok=True)
    os.makedirs(output_absorption_images3, exist_ok=True)
    os.makedirs(qr_matrices, exist_ok=True)


def solve_pvb_sweep(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
    player_temperature=None,
    max_attempts=100,
):
    N = n - 1
    inner_n = n - 2
    radius = compute_radius(n)
    border_cases = get_border_cases(N)

    validate_pvb_geometry(n, inner_n, radius)
    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)

    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    real_pmf = None
    if real_pmf_path is not None:
        real_pmf = np.load(real_pmf_path)[: num_steps + 1].astype(float)
        if real_pmf.sum() > 0:
            real_pmf /= real_pmf.sum()

    w_new_list = []
    strategy_snapshots = []
    mean_times = []
    fit_scores = []
    symmetric_start = build_symmetric_legacy_start(inner_n, n)
    solver_cache = build_pvb_solver_cache(n, inner_n, radius)
    for epsilon in tqdm(epsilon_values, desc="Solving equations"):
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            if attempts == 1:
                starting_params = symmetric_start
            else:
                starting_params = np.random.random(inner_n ** 2) * (inner_n - 2) ** 2
            w_new, _, _, message = fsolve(
                lambda w: prepare_equations_cached(
                    w,
                    epsilon,
                    n,
                    inner_n,
                    radius,
                    border_cases,
                    solver_cache,
                    player_temperature=player_temperature,
                ),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        p1_flat, q1_flat, v_flat = compute_state_values_cached(
            w_new,
            epsilon,
            n,
            inner_n,
            radius,
            border_cases,
            solver_cache,
            player_temperature=player_temperature,
        )

        p1_matrix = np.reshape(p1_flat, (inner_n, inner_n))
        q1_matrix = np.reshape(q1_flat, (inner_n, inner_n))

        qr_optimal, probability_optimal = make_prob_matrix(
            N,
            np.pad(p1_matrix, pad_width=1, mode="constant", constant_values=0).T,
            np.pad(q1_matrix, pad_width=1, mode="constant", constant_values=0).T,
        )
        mean_time, state_mean_times = find_mean_time_banded(probability_optimal, N - 1)
        mean_times.append(mean_time)
        np.save(qr_matrices + f"qr_{epsilon:.2f}", qr_optimal)

        fit_score = None
        if real_pmf is not None:
            _, prob, _ = model_pvp(N, qr_optimal, num_steps=num_steps)
            prob = np.asarray(prob, dtype=float)
            if prob.sum() > 0:
                prob /= prob.sum()
            fit_score = float(jensenshannon(real_pmf, prob))
        fit_scores.append(fit_score)

        strategy_snapshots.append(
            {
                "epsilon": float(epsilon),
                "p1s": p1_flat,
                "q1s": q1_flat,
                "vs": v_flat,
                "w": w_new,
                "mean_time": float(mean_time),
                "state_mean_times": state_mean_times,
                "jsd": fit_score,
            }
        )

    with open(output_duration + "duration.txt", "w") as file:
        for mean_time in mean_times:
            file.write(f"{mean_time} ")

    if real_pmf is not None:
        with open(output_duration + "jsd.txt", "w") as file:
            for score in fit_scores:
                file.write(f"{score} ")

    return {
        "n": n,
        "N": N,
        "inner_n": inner_n,
        "radius": radius,
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": w_new_list,
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "player_temperature": player_temperature,
    }

def compute_geometry_center_score(global_index, global_n):
    max_border_distance = max(1.0, global_n // 2)
    max_center_distance = max(1.0, global_n - 2)

    border_score = compute_distance_to_border(global_index, global_n) / max_border_distance
    center_score = 1.0 - (compute_distance_to_center(global_index, global_n) / max_center_distance)
    return 0.7 * border_score + 0.3 * center_score


def compute_directional_safety_score(global_index, direction, global_n):
    next_global_index = move_global_index(global_index, global_n, direction)
    next_row, next_col = get_row_col(next_global_index, global_n)
    if not (1 <= next_row <= global_n - 2 and 1 <= next_col <= global_n - 2):
        return -1.0

    current_border_distance = compute_distance_to_border(global_index, global_n)
    current_center_distance = compute_distance_to_center(global_index, global_n)
    next_border_distance = compute_distance_to_border(next_global_index, global_n)
    next_center_distance = compute_distance_to_center(next_global_index, global_n)

    border_delta = next_border_distance - current_border_distance
    center_delta = current_center_distance - next_center_distance
    geometry_score = compute_geometry_center_score(next_global_index, global_n)

    return geometry_score + 0.45 * border_delta + 0.20 * center_delta


def compute_geometric_center_probability(global_index, epsilon, global_n):
    strategy_zero_score = (
        compute_directional_safety_score(global_index, "up", global_n)
        + compute_directional_safety_score(global_index, "right", global_n)
    ) / 2.0
    strategy_one_score = (
        compute_directional_safety_score(global_index, "down", global_n)
        + compute_directional_safety_score(global_index, "left", global_n)
    ) / 2.0

    # In make_prob_matrix, p1 is the probability of the second pure center strategy:
    # "down/left". A larger safety score for "up/right" therefore must decrease p1.
    return expit(epsilon * (strategy_one_score - strategy_zero_score))


def build_geometric_center_strategy(global_n, epsilon):
    strategy_center = np.zeros((global_n, global_n), dtype=float)
    for row in range(1, global_n - 1):
        for col in range(1, global_n - 1):
            global_index = row * global_n + col
            strategy_center[row, col] = compute_geometric_center_probability(
                global_index,
                epsilon,
                global_n,
            )
    return strategy_center
