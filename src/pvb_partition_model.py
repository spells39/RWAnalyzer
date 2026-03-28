import os

import numpy as np
import scipy.linalg
from scipy.spatial.distance import jensenshannon
from scipy.special import expit, logit
from scipy.optimize import fsolve
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp


def find_max(func, game, temperature=None):
    value_if_strategy_0 = func(0.0, game)
    value_if_strategy_1 = func(1.0, game)

    if temperature is None:
        if value_if_strategy_0 >= value_if_strategy_1:
            return value_if_strategy_0, 0.0
        return value_if_strategy_1, 1.0

    delta = (value_if_strategy_1 - value_if_strategy_0) / temperature
    p = 1.0 / (1.0 + np.exp(-delta))
    value = func(p, game)
    return value, p


def get_win(p, game):
    return (p / 2.0) * (game[0, 0] + game[0, 1]) + ((1.0 - p) / 2.0) * (game[1, 0] + game[1, 1])


def get_value(game, temperature=None):
    return find_max(get_win, game, temperature=temperature)


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


def compute_epsilon_border(inner_index, epsilon, direction, radius, global_n, inner_n):
    global_index = inner_n_to_global_N(inner_index, inner_n, global_n)
    cur_distance = compute_distance_to_center(global_index, global_n)
    distance_to_border = compute_distance_to_border(global_index, global_n)

    if cur_distance <= radius:
        return 0.0

    new_global_index = move_global_index(global_index, global_n, direction)
    new_distance_to_border = compute_distance_to_border(new_global_index, global_n)
    if new_distance_to_border < distance_to_border:
        return -epsilon
    return epsilon


def get_game(index, w, epsilon, radius, n, inner_n, border_cases):
    game = np.zeros((2, 2))
    game[0, 0] = compute_a11(index, w, epsilon, radius, n, inner_n, border_cases)
    game[0, 1] = compute_a12(index, w, epsilon, radius, n, inner_n, border_cases)
    game[1, 0] = compute_a21(index, w, epsilon, radius, n, inner_n, border_cases)
    game[1, 1] = compute_a22(index, w, epsilon, radius, n, inner_n, border_cases)
    return game


def compute_a11(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index - n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - n, inner_n, n)
    adjusted_epsilon = compute_epsilon_border(next_inner_index, epsilon, "up", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def compute_a21(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + n) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + n, inner_n, n)
    adjusted_epsilon = compute_epsilon_border(next_inner_index, epsilon, "down", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def compute_a12(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index + 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index + 1, inner_n, n)
    adjusted_epsilon = compute_epsilon_border(next_inner_index, epsilon, "right", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def compute_a22(index, w, epsilon, radius, n, inner_n, border_cases):
    if (index - 1) in border_cases:
        return 1.0
    next_inner_index = global_N_to_inner_n(index - 1, inner_n, n)
    adjusted_epsilon = compute_epsilon_border(next_inner_index, epsilon, "left", radius, n, inner_n)
    return w[next_inner_index] + 1.0 + adjusted_epsilon


def prepare_equations(w, epsilon, n, inner_n, radius, border_cases, player_temperature=None):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, _ = get_value(game_mx, temperature=player_temperature)
        eqs[i] = w[i] - v
    return tuple(eqs)


def compute_state_values(w, epsilon, n, inner_n, radius, border_cases, player_temperature=None):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, p1 = get_value(game_mx, temperature=player_temperature)
        p1s.append(p1)
        q1s.append(0.5)
        vs.append(v)
    return np.array(p1s), np.array(q1s), np.array(vs)


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
        current_border_distance = compute_distance_to_border(global_index, n)
        current_center_distance = compute_distance_to_center(global_index, n)

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

            next_global_index = next_row * n + next_col
            next_border_distance = compute_distance_to_border(next_global_index, n)
            epsilon_sign = np.sign(
                compute_epsilon_border(inner_index, 1.0, direction, radius, n, inner_n)
            )

            if current_center_distance <= radius and epsilon_sign != 0.0:
                raise AssertionError("Center radius should suppress epsilon.")

            if current_center_distance > radius:
                expected_sign = -1.0 if next_border_distance < current_border_distance else 1.0
                if epsilon_sign != expected_sign:
                    raise AssertionError(
                        f"Unexpected epsilon sign for state={inner_index}, direction={direction}."
                    )


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

    w_new_list = []
    strategy_snapshots = []
    mean_times = []

    for epsilon in tqdm(epsilon_values, desc="Solving equations"):
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            starting_params = np.random.random(inner_n ** 2) * (inner_n - 2) ** 2
            w_new, _, _, message = fsolve(
                lambda w: prepare_equations(
                    w,
                    epsilon,
                    n,
                    inner_n,
                    radius,
                    border_cases,
                    player_temperature=player_temperature,
                ),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)

        p1_flat, q1_flat, v_flat = compute_state_values(
            w_new,
            epsilon,
            n,
            inner_n,
            radius,
            border_cases,
            player_temperature=player_temperature,
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
        "epsilon_values": np.array(epsilon_values),
        "w_new_list": w_new_list,
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times),
        "player_temperature": player_temperature,
    }


def load_empirical_center_strategy(strategy_path):
    return np.load(strategy_path)


def scale_center_strategy_logit(strategy_center, epsilon, clip=1e-4):
    clipped = np.clip(strategy_center, clip, 1.0 - clip)
    return expit(logit(clipped) * epsilon)


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


def compute_duration_distribution_from_strategies(N, strategy_center, strategy_border, num_steps=999):
    qr, probability_optimal = make_prob_matrix(N, strategy_center, strategy_border)
    _, prob, _ = model_pvp(N, qr, num_steps=num_steps)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() > 0:
        prob /= prob.sum()
    return qr, probability_optimal, prob


def solve_pvb_empirical_strategy_sweep(
    epsilon_values,
    strategy_center_path,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
):
    strategy_center_base = load_empirical_center_strategy(strategy_center_path)
    N = strategy_center_base.shape[0] - 1
    inner_n = N - 1
    n = N + 1

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
    strategy_border = np.full_like(strategy_center_base, 0.5, dtype=float)

    for epsilon in tqdm(epsilon_values, desc="Scaling empirical center strategy"):
        strategy_center = scale_center_strategy_logit(strategy_center_base, epsilon)
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
                "mean_time": mean_time,
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
        "radius": None,
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "strategy_center_base": strategy_center_base,
        "mode": "empirical-logit",
    }


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


def solve_pvb_geometric_strategy_sweep(
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
):
    n = 17
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
    strategy_border = np.full((n, n), 0.5, dtype=float)

    for epsilon in tqdm(epsilon_values, desc="Building geometric PvB center strategy"):
        strategy_center = build_geometric_center_strategy(n, epsilon)
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
                "mean_time": mean_time,
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
        "radius": compute_radius(n),
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "mode": "geometric-center",
    }
