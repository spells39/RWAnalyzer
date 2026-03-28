import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import expit
from tqdm import tqdm

from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp
from quarter_partition_utils import (
    PURE_CENTER_DL,
    PURE_CENTER_UR,
    compute_distance_to_border,
    compute_distance_to_center,
    ensure_output_dirs,
    find_mean_time_banded,
    is_inner_state,
    move_global_index,
    preferred_center_directions,
)


def compute_center_direction_score(global_index, direction, global_n):
    next_global_index = move_global_index(global_index, global_n, direction)
    if not is_inner_state(next_global_index, global_n):
        return -3.0

    preferred = preferred_center_directions(global_index, global_n)
    alignment = 1.0 if direction in preferred else -1.0
    border_delta = compute_distance_to_border(next_global_index, global_n) - compute_distance_to_border(global_index, global_n)
    center_delta = compute_distance_to_center(global_index, global_n) - compute_distance_to_center(next_global_index, global_n)

    return 0.9 * alignment + 0.35 * border_delta + 0.10 * center_delta


def compute_center_probability_dl(global_index, epsilon, global_n):
    score_ur = np.mean([compute_center_direction_score(global_index, direction, global_n) for direction in PURE_CENTER_UR])
    score_dl = np.mean([compute_center_direction_score(global_index, direction, global_n) for direction in PURE_CENTER_DL])
    return expit(epsilon * (score_dl - score_ur))


def build_center_strategy(global_n, epsilon):
    strategy_center = np.full((global_n, global_n), 0.5, dtype=float)
    for row in range(1, global_n - 1):
        for col in range(1, global_n - 1):
            global_index = row * global_n + col
            strategy_center[row, col] = compute_center_probability_dl(global_index, epsilon, global_n)
    return strategy_center


def compute_duration_distribution_from_strategies(N, strategy_center, strategy_border, num_steps=999):
    qr, probability_optimal = make_prob_matrix(N, strategy_center, strategy_border)
    _, prob, _ = model_pvp(N, qr, num_steps=num_steps)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() > 0:
        prob /= prob.sum()
    return qr, probability_optimal, prob


def solve_pvb_quarter_sweep(
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
    n=17,
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
    strategy_border = np.full((n, n), 0.5, dtype=float)

    for epsilon in tqdm(epsilon_values, desc="Building quarter-based PvB center strategy"):
        strategy_center = build_center_strategy(n, epsilon)
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

    if real_pmf is not None:
        with open(output_duration + "jsd.txt", "w") as file:
            for score in fit_scores:
                file.write(f"{score} ")

    return {
        "n": n,
        "N": N,
        "inner_n": inner_n,
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "mode": "quarter-geometric-center",
    }
