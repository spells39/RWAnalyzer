import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm

from get_border_cases import get_border_cases
from make_prob_matrix import make_prob_matrix
from quarter_partition_utils import (
    ensure_output_dirs,
    find_mean_time_banded,
    global_N_to_inner_n,
    inner_n_to_global_N,
    preferred_border_directions,
)


def find_max(func, game):
    if func(0.0, game) < func(1.0, game):
        return func(0.0, game), 0.0
    return func(1.0, game), 1.0


def get_win(q_vertical, game):
    return (q_vertical / 2.0) * (game[0, 0] + game[1, 0]) + ((1.0 - q_vertical) / 2.0) * (game[0, 1] + game[1, 1])


def get_value(game):
    return find_max(get_win, game)


def compute_quarter_epsilon(global_index, epsilon, direction, global_n):
    preferred = preferred_border_directions(global_index, global_n)
    return -epsilon if direction in preferred else epsilon


def get_game(index, w, epsilon, n, inner_n, border_cases):
    game = np.zeros((2, 2))
    game[0, 0] = compute_a11(index, w, epsilon, n, inner_n, border_cases)
    game[0, 1] = compute_a12(index, w, epsilon, n, inner_n, border_cases)
    game[1, 0] = compute_a21(index, w, epsilon, n, inner_n, border_cases)
    game[1, 1] = compute_a22(index, w, epsilon, n, inner_n, border_cases)
    return game


def compute_a11(index, w, epsilon, n, inner_n, border_cases):
    if (index - n) in border_cases:
        return 1.0
    next_index = index - n
    next_inner_index = global_N_to_inner_n(next_index, inner_n, n)
    adjusted = compute_quarter_epsilon(next_index, epsilon, "up", n)
    return w[next_inner_index] + 1.0 + adjusted


def compute_a21(index, w, epsilon, n, inner_n, border_cases):
    if (index + n) in border_cases:
        return 1.0
    next_index = index + n
    next_inner_index = global_N_to_inner_n(next_index, inner_n, n)
    adjusted = compute_quarter_epsilon(next_index, epsilon, "down", n)
    return w[next_inner_index] + 1.0 + adjusted


def compute_a12(index, w, epsilon, n, inner_n, border_cases):
    if (index + 1) in border_cases:
        return 1.0
    next_index = index + 1
    next_inner_index = global_N_to_inner_n(next_index, inner_n, n)
    adjusted = compute_quarter_epsilon(next_index, epsilon, "right", n)
    return w[next_inner_index] + 1.0 + adjusted


def compute_a22(index, w, epsilon, n, inner_n, border_cases):
    if (index - 1) in border_cases:
        return 1.0
    next_index = index - 1
    next_inner_index = global_N_to_inner_n(next_index, inner_n, n)
    adjusted = compute_quarter_epsilon(next_index, epsilon, "left", n)
    return w[next_inner_index] + 1.0 + adjusted


def prepare_equations(w, epsilon, n, inner_n, border_cases):
    eqs = np.zeros(len(w))
    for i in range(len(w)):
        index = inner_n_to_global_N(i, inner_n, n)
        value, _ = get_value(get_game(index, w, epsilon, n, inner_n, border_cases))
        eqs[i] = w[i] - value
    return tuple(eqs)


def compute_state_values(w, epsilon, n, inner_n, border_cases):
    p1s = np.full(inner_n ** 2, 0.5, dtype=float)
    q1s = np.zeros(inner_n ** 2, dtype=float)
    for i in range(inner_n ** 2):
        index = inner_n_to_global_N(i, inner_n, n)
        _, q_vertical = get_value(get_game(index, w, epsilon, n, inner_n, border_cases))
        q1s[i] = q_vertical
    return p1s, q1s


def solve_bvp_quarter_sweep(
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
    border_cases = get_border_cases(N)

    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)

    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    w_new_list = []
    strategy_snapshots = []
    mean_times = []

    for epsilon in tqdm(epsilon_values, desc="Solving quarter-based BvP equations"):
        message = ""
        attempts = 0
        while message != "The solution converged.":
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to converge for epsilon={epsilon:.3f}: {message}")
            starting_params = np.random.random(inner_n ** 2) * (inner_n - 1) ** 2
            w_new, _, _, message = fsolve(
                lambda w: prepare_equations(w, epsilon, n, inner_n, border_cases),
                tuple(starting_params),
                full_output=True,
            )

        w_new_list.append(w_new)
        p1s, q1s = compute_state_values(w_new, epsilon, n, inner_n, border_cases)

        qr_optimal, probability_optimal = make_prob_matrix(
            N,
            np.pad(np.reshape(p1s, (inner_n, inner_n)), pad_width=1, mode="constant", constant_values=0).T,
            np.pad(np.reshape(q1s, (inner_n, inner_n)), pad_width=1, mode="constant", constant_values=0).T,
        )
        mean_time, state_mean_times = find_mean_time_banded(probability_optimal, N - 1)
        mean_times.append(mean_time)
        np.save(qr_matrices + f"qr_{epsilon:.2f}", qr_optimal)

        strategy_snapshots.append(
            {
                "epsilon": float(epsilon),
                "p1s": p1s,
                "q1s": q1s,
                "vs": state_mean_times,
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
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": w_new_list,
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "mode": "quarter-game-border",
    }
