import os

import numpy as np
import scipy.linalg
from scipy.optimize import fsolve
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp


DEFAULT_BVP_SMOOTH_ALPHA = 3.0
PURE_BORDER_VERTICAL = ("up", "down")
PURE_BORDER_HORIZONTAL = ("right", "left")

OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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


def get_row_col(global_index, global_n):
    return global_index // global_n, global_index % global_n


def move_global_index(global_index, global_n, direction):
    row, col = get_row_col(global_index, global_n)
    dr, dc = OFFSETS[direction]
    return (row + dr) * global_n + (col + dc)


def is_inner_state(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    return 1 <= row <= global_n - 2 and 1 <= col <= global_n - 2


def compute_distance_to_center(global_index, global_n):
    center_row = global_n // 2
    center_col = global_n // 2
    row, col = get_row_col(global_index, global_n)
    return abs(row - center_row) + abs(col - center_col)


def compute_distance_to_border(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    distances = [row, global_n - 1 - row, col, global_n - 1 - col]
    return min(distances)


def get_quarter(global_index, global_n):
    row, col = get_row_col(global_index, global_n)
    center_row = global_n // 2
    center_col = global_n // 2
    if row == center_row and col == center_col:
        return 0
    if row == center_row:
        return 8 if col > center_col else 4
    if col == center_col:
        return 2 if row < center_row else 6
    if row < center_row:
        return 1 if col > center_col else 3
    return 7 if col > center_col else 5


def preferred_border_directions(global_index, global_n):
    quarter = get_quarter(global_index, global_n)
    return {
        0: set(),
        1: {"up", "right"},
        2: {"up"},
        3: {"up", "left"},
        4: {"left"},
        5: {"down", "left"},
        6: {"down"},
        7: {"down", "right"},
        8: {"right"},
    }[quarter]


def compute_radius(global_n):
    return global_n // 4


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


def _legacy_compute_epsilon_border(inner_index, epsilon, direction, radius, global_n, inner_n):
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
    adjusted_epsilon = _legacy_compute_epsilon_border(next_inner_index, epsilon, "up", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a21(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + n, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(next_inner_index, epsilon, "down", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a12(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(next_inner_index, epsilon, "right", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_compute_a22(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index - 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - 1, inner_n, n)
    adjusted_epsilon = _legacy_compute_epsilon_border(next_inner_index, epsilon, "left", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def _legacy_prepare_equations(w, epsilon, n, inner_n, radius, border_cases):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, _ = get_value(game_mx)
        eqs[i] = w[i] - v
    return tuple(eqs)


def _legacy_compute_state_values(w, epsilon, n, inner_n, radius, border_cases):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, q1 = get_value(game_mx)
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

    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)
    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    w_new_list = []
    strategy_snapshots = []
    mean_times = []

    for epsilon in tqdm(epsilon_values, desc="Solving legacy equations"):
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
        p1_flat, q1_flat, v_flat = _legacy_compute_state_values(w_new, epsilon, n, inner_n, radius, border_cases)
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
            }
        )

    with open(output_duration + "duration.txt", "w") as file:
        for mean_time in mean_times:
            file.write(f"{mean_time} ")

    return {
        "n": n,
        "N": N,
        "inner_n": inner_n,
        "radius": radius,
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": w_new_list,
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "mode": "border-legacy",
    }


def smooth_best_response_vertical_probability(game, smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA):
    value_vertical = get_win(1.0, game)
    value_horizontal = get_win(0.0, game)
    return expit(smooth_alpha * (value_vertical - value_horizontal))


def _smooth_prepare_equations(w, epsilon, n, inner_n, radius, border_cases, smooth_alpha):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        q_vertical = smooth_best_response_vertical_probability(game_mx, smooth_alpha=smooth_alpha)
        v = get_win(q_vertical, game_mx)
        eqs[i] = w[i] - v
    return tuple(eqs)


def _smooth_compute_state_values(w, epsilon, n, inner_n, radius, border_cases, smooth_alpha):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = _legacy_get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        q_vertical = smooth_best_response_vertical_probability(game_mx, smooth_alpha=smooth_alpha)
        v = get_win(q_vertical, game_mx)
        p1s.append(0.5)
        q1s.append(q_vertical)
        vs.append(v)
    return np.array(p1s), np.array(q1s), np.array(vs)


def solve_bvp_sweep(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
    smooth_alpha=DEFAULT_BVP_SMOOTH_ALPHA,
    **kwargs,
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
    fit_scores = []

    real_pmf = None
    if real_pmf_path is not None:
        real_pmf = np.load(real_pmf_path)[: num_steps + 1].astype(float)
        if real_pmf.sum() > 0:
            real_pmf /= real_pmf.sum()

    for epsilon in tqdm(epsilon_values, desc="Solving smooth BvP equations"):
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > kwargs.get("max_attempts", 100):
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            starting_params = np.random.random(inner_n ** 2) * (inner_n - 1) ** 2
            w_new, _, _, message = fsolve(
                lambda w: _smooth_prepare_equations(w, epsilon, n, inner_n, radius, border_cases, smooth_alpha),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        p1_flat, q1_flat, v_flat = _smooth_compute_state_values(
            w_new,
            epsilon,
            n,
            inner_n,
            radius,
            border_cases,
            smooth_alpha,
        )
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

        fit_score = None
        if real_pmf_path is not None:
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
        "smooth_alpha": float(smooth_alpha),
        "mode": "border-smooth-best-response",
    }
