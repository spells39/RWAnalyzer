import os
import time

import numpy as np
import scipy.linalg
from scipy.optimize import fsolve
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp
from quarter_partition_utils import (
    PURE_BORDER_HORIZONTAL,
    PURE_BORDER_VERTICAL,
    compute_distance_to_border,
    compute_distance_to_center,
    ensure_output_dirs,
    find_mean_time_banded,
    get_row_col,
    is_inner_state,
    move_global_index,
    preferred_border_directions,
)


# Legacy BvP solver is preserved below as *_legacy helpers.

DEFAULT_BVP_SMOOTH_ALPHA = 4.0
DEFAULT_BVP_FEATURE_TAU = 0.15
DEFAULT_BVP_BASELINE_BLEND_TAU = 0.03


def find_max(func, game):
    value_if_strategy_0 = func(0.0, game)
    value_if_strategy_1 = func(1.0, game)
    if value_if_strategy_0 < value_if_strategy_1:
        return value_if_strategy_0, 0.0
    return value_if_strategy_1, 1.0


def get_win(p, game):
    return (p / 2.0) * (game[0, 0] + game[1, 0]) + ((1.0 - p) / 2.0) * (game[0, 1] + game[1, 1])


def get_value(game):
    return find_max(get_win, game)


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


def compute_radius(global_n):
    return global_n // 4


def make_banded_matrix(A, N):
    banded_matrix = np.zeros((2 * N + 1, N ** 2), dtype=np.float64)
    for i in [-N, -1, 0, 1, N]:
        diagonal = np.diagonal(A, -i)
        if i < 0:
            banded_matrix[i + N, -i:] = diagonal
        elif i > 0:
            banded_matrix[i + N, :-i] = diagonal
        else:
            banded_matrix[i + N, :] = diagonal
    return banded_matrix


def _legacy_compute_direction_bias(inner_index, epsilon, direction, radius, global_n, inner_n):
    global_index = inner_n_to_global_N(inner_index, inner_n, global_n)
    cur_distance = compute_distance_to_center(global_index, global_n)
    distance_to_border = compute_distance_to_border(global_index, global_n)

    if cur_distance <= radius:
        return 0.0

    new_global_index = move_global_index(global_index, global_n, direction)
    new_distance_to_border = compute_distance_to_border(new_global_index, global_n)
    if new_distance_to_border < distance_to_border:
        return epsilon
    return -epsilon


def _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases):
    game = np.zeros((2, 2))
    game[0, 0] = _legacy_compute_a11(index, w, epsilon, radius, n, inner_n, border_cases)
    game[0, 1] = _legacy_compute_a12(index, w, epsilon, radius, n, inner_n, border_cases)
    game[1, 0] = _legacy_compute_a21(index, w, epsilon, radius, n, inner_n, border_cases)
    game[1, 1] = _legacy_compute_a22(index, w, epsilon, radius, n, inner_n, border_cases)
    return game


def _legacy_compute_a11(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index - n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - n, inner_n, n)
    adjusted_epsilon = _legacy_compute_direction_bias(next_inner_index, epsilon, "up", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a21(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + n, inner_n, n)
    adjusted_epsilon = _legacy_compute_direction_bias(next_inner_index, epsilon, "down", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a12(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_direction_bias(next_inner_index, epsilon, "right", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a22(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index - 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_direction_bias(next_inner_index, epsilon, "left", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_prepare_equations(w, epsilon, n, inner_n, radius, border_cases):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        value, _ = get_value(_legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases))
        eqs[i] = w[i] - value
    return tuple(eqs)


def _legacy_compute_state_values(w, epsilon, n, inner_n, radius, border_cases):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        value, q_vertical = get_value(_legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases))
        p1s.append(0.5)
        q1s.append(q_vertical)
        vs.append(value)
    return np.array(p1s), np.array(q1s), np.array(vs)


def solve_bvp_quarter_sweep_legacy(
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    n=17,
    max_attempts=100,
):
    N = n - 1
    inner_n = n - 2
    radius = compute_radius(n)
    border_cases = get_border_cases(N)

    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)
    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    w_new_list = []
    strategy_snapshots = []
    mean_times = []
    solve_times = []
    attempt_counts = []

    for epsilon in tqdm(epsilon_values, desc="Solving legacy BvP equations"):
        started_at = time.perf_counter()
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            starting_params = np.random.random(inner_n ** 2) * (inner_n - 1) ** 2
            w_new, _, _, message = fsolve(
                lambda w: _legacy_prepare_equations(w, epsilon, n, inner_n, radius, border_cases),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        attempt_counts.append(attempts)
        solve_times.append(time.perf_counter() - started_at)
        p1_flat, q1_flat, v_flat = _legacy_compute_state_values(w_new, epsilon, n, inner_n, radius, border_cases)
        qr_optimal, probability_optimal = make_prob_matrix(
            N,
            np.pad(np.reshape(p1_flat, (inner_n, inner_n)), pad_width=1, mode="constant", constant_values=0).T,
            1.0
            - np.pad(np.reshape(q1_flat, (inner_n, inner_n)), pad_width=1, mode="constant", constant_values=0).T,
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
            }
        )

    with open(output_duration + "duration.txt", "w") as file:
        for mean_time in mean_times:
            file.write(f"{mean_time} ")

    with open(output_duration + "absorption_time.txt", "w") as file:
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
        "mode": "quarter-border-legacy",
    }


def is_border_state(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    return row == 0 or col == 0 or row == global_n - 1 or col == global_n - 1


def compute_border_direction_score(global_index, direction, global_n):
    next_global_index = move_global_index(global_index, global_n, direction)
    if not is_inner_state(next_global_index, global_n) and not is_border_state(next_global_index, global_n):
        return -1.0

    current_border_distance = compute_distance_to_border(global_index, global_n)
    next_border_distance = compute_distance_to_border(next_global_index, global_n)
    return current_border_distance - next_border_distance


def compute_border_axis_score(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    center = global_n // 2

    top = float(row)
    bottom = float(global_n - 1 - row)
    left = float(col)
    right = float(global_n - 1 - col)
    nearest_vertical = min(top, bottom)
    nearest_horizontal = min(left, right)

    axis_min = 1.0 / (nearest_vertical + 1.0) - 1.0 / (nearest_horizontal + 1.0)
    axis_sum = 1.0 / (top + 1.0) + 1.0 / (bottom + 1.0) - 1.0 / (left + 1.0) - 1.0 / (right + 1.0)
    center_row = np.exp(-((row - center) / 2.0) ** 2)
    center_col = -np.exp(-((col - center) / 2.0) ** 2)
    near_top_bottom = np.exp(-nearest_vertical / 2.0)
    near_left_right = -np.exp(-nearest_horizontal / 2.0)
    diagonal_balance = (abs(row - center) - abs(col - center)) / max(1.0, global_n / 2.0)

    # Coefficients were selected on geometry-only features so the map is no longer a
    # hard quarter template, but every term still depends on nearest-border geometry.
    return (
        -0.056
        + 0.358 * axis_min
        + 0.644 * axis_sum
        + 0.259 * center_row
        + 0.310 * center_col
        + 0.553 * near_top_bottom
        + 0.441 * near_left_right
        - 0.789 * diagonal_balance
    )


def compute_border_baseline_probability(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    center = global_n // 2
    vertical_offset = abs(row - center)
    horizontal_offset = abs(col - center)

    if vertical_offset > horizontal_offset:
        return 1.0
    if horizontal_offset > vertical_offset:
        return 0.0
    return 0.5


def compute_border_probability_vertical(global_index, epsilon, global_n, smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA):
    baseline_probability = compute_border_baseline_probability(global_index, global_n)
    if epsilon <= 0.0:
        return baseline_probability

    # Keep epsilon=0 as the exact quarter baseline, but avoid the old fast
    # exponential saturation where most positive epsilons produced almost the
    # same strategy map.
    activation = smooth_alpha * float(epsilon)
    feature_probability = float(expit(activation * compute_border_axis_score(global_index, global_n)))
    blend_weight = 1.0 - np.exp(-float(epsilon) / DEFAULT_BVP_BASELINE_BLEND_TAU)
    return float((1.0 - blend_weight) * baseline_probability + blend_weight * feature_probability)


def compute_border_probability_horizontal(global_index, epsilon, global_n, smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA):
    return compute_border_probability_vertical(global_index, epsilon, global_n, smooth_alpha=smooth_alpha)


def build_border_strategy(global_n, epsilon, smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA):
    strategy_border = np.full((global_n, global_n), 0.5, dtype=float)
    for row in range(1, global_n - 1):
        for col in range(1, global_n - 1):
            global_index = row * global_n + col
            strategy_border[row, col] = compute_border_probability_vertical(
                global_index,
                epsilon,
                global_n,
                smooth_alpha=smooth_alpha,
            )
    return strategy_border


def compute_duration_distribution_from_strategies(N, strategy_center, strategy_border, num_steps=999):
    # strategy_border is stored and plotted canonically as P(up/down).
    # make_prob_matrix expects P(right/left), so convert only at the transition
    # layer. This keeps the images readable while preserving game semantics.
    qr, probability_optimal = make_prob_matrix(N, strategy_center, 1.0 - strategy_border)
    _, prob, _ = model_pvp(N, qr, num_steps=num_steps)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() > 0:
        prob /= prob.sum()
    return qr, probability_optimal, prob


def solve_bvp_quarter_sweep(
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
    n=17,
    smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA,
):
    N = n - 1
    inner_n = n - 2

    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)
    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    real_pmf = None
    if real_pmf_path is not None:
        real_pmf = np.load(real_pmf_path)[: num_steps + 1].astype(float)
        if real_pmf.sum() > 0:
            real_pmf /= real_pmf.sum()

    strategy_snapshots = []
    mean_times = []
    fit_scores = []
    strategy_center = np.full((n, n), 0.5, dtype=float)

    for epsilon in tqdm(epsilon_values, desc="Building smooth quarter-based BvP border strategy"):
        strategy_border = build_border_strategy(n, epsilon, smooth_alpha=smooth_alpha)
        qr_optimal, probability_optimal, prob = compute_duration_distribution_from_strategies(
            N,
            strategy_center,
            strategy_border,
            num_steps=num_steps,
        )
        np.save(qr_matrices + f"qr_{epsilon:.2f}", qr_optimal)

        mean_time, state_mean_times = find_mean_time_banded(probability_optimal, N - 1)
        mean_times.append(mean_time)

        fit_score = None
        if real_pmf is not None:
            fit_score = float(jensenshannon(real_pmf, prob))
        fit_scores.append(fit_score)

        strategy_snapshots.append(
            {
                "epsilon": float(epsilon),
                "p1s": strategy_center[1:-1, 1:-1].reshape(inner_n ** 2),
                "q1s": strategy_border[1:-1, 1:-1].reshape(inner_n ** 2),
                "vs": state_mean_times,
                "w": None,
                "mean_time": float(mean_time),
                "jsd": fit_score,
            }
        )

    with open(output_duration + "duration.txt", "w") as file:
        for mean_time in mean_times:
            file.write(f"{mean_time} ")

    with open(output_duration + "absorption_time.txt", "w") as file:
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
        "radius": compute_radius(n),
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "smooth_alpha": float(smooth_alpha),
        "mode": "quarter-border-smooth",
    }


def solve_bvp_sweep(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    **kwargs,
):
    return solve_bvp_quarter_sweep(
        epsilon_values=epsilon_values,
        output_duration=output_duration,
        output_absorption_images1=output_absorption_images1,
        output_absorption_images2=output_absorption_images2,
        output_absorption_images3=output_absorption_images3,
        qr_matrices=qr_matrices,
        n=n,
        **kwargs,
    )
