import os

import numpy as np
import scipy.linalg
from scipy.optimize import fsolve
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix


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


def compute_distance_to_center(global_index, global_n):
    center_row = global_n // 2
    center_col = global_n // 2
    row = global_index // global_n
    col = global_index % global_n
    return abs(row - center_row) + abs(col - center_col)


def compute_distance_to_border(global_index, global_n):
    row = global_index // global_n
    col = global_index % global_n
    distances = [row, global_n - 1 - row, col, global_n - 1 - col]
    return min(distances)


def compute_radius(global_n):
    return global_n // 4


def move_global_index(global_index, direction, global_n):
    if direction == "up":
        return global_index - global_n
    if direction == "down":
        return global_index + global_n
    if direction == "left":
        return global_index - 1
    if direction == "right":
        return global_index + 1
    raise ValueError("Unknown direction")


def compute_epsilon_border(inner_index, epsilon, direction, radius, global_n, inner_n):
    global_index = inner_n_to_global_N(inner_index, inner_n, global_n)
    cur_distance = compute_distance_to_center(global_index, global_n)
    distance_to_border = compute_distance_to_border(global_index, global_n)

    if cur_distance <= radius:
        return 0.0

    new_global_index = move_global_index(global_index, direction, global_n)
    new_distance_to_border = compute_distance_to_border(new_global_index, global_n)
    if new_distance_to_border < distance_to_border:
        return epsilon
    return -epsilon


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


def prepare_equations(w, epsilon, n, inner_n, radius, border_cases):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, _ = get_value(game_mx)
        eqs[i] = w[i] - v
    return tuple(eqs)


def compute_state_values(w, epsilon, n, inner_n, radius, border_cases):
    p1s = []
    q1s = []
    vs = []
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        game_mx = get_game(index, w, epsilon, radius, n, inner_n, border_cases)
        v, q1 = get_value(game_mx)
        p1s.append(0.5)
        q1s.append(q1)
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


def ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices):
    os.makedirs(output_absorption_images1, exist_ok=True)
    os.makedirs(output_absorption_images2, exist_ok=True)
    os.makedirs(output_absorption_images3, exist_ok=True)
    os.makedirs(qr_matrices, exist_ok=True)


def solve_bvp_sweep(
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

    for epsilon in tqdm(epsilon_values, desc="Solving equations"):
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")

            starting_params = np.random.random(inner_n ** 2) * (inner_n - 1) ** 2
            w_new, _, _, message = fsolve(
                lambda w: prepare_equations(w, epsilon, n, inner_n, radius, border_cases),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        p1_flat, q1_flat, v_flat = compute_state_values(w_new, epsilon, n, inner_n, radius, border_cases)
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
    }
