import os

import numpy as np
import scipy.linalg
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp
from pvb_partition_model import build_geometric_center_strategy


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


def find_mean_time_banded(A, N):
    banded_matrix = make_banded_matrix(np.eye(A.shape[0]) - A, N)
    mean_times = scipy.linalg.solve_banded((N, N), banded_matrix, np.ones(banded_matrix.shape[1]))
    return mean_times[N * (N // 2) + N // 2], mean_times


def ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices):
    os.makedirs(output_absorption_images1, exist_ok=True)
    os.makedirs(output_absorption_images2, exist_ok=True)
    os.makedirs(output_absorption_images3, exist_ok=True)
    os.makedirs(qr_matrices, exist_ok=True)


def compute_duration_distribution_from_strategies(N, strategy_center, strategy_border, num_steps=9999):
    qr, probability_optimal = make_prob_matrix(N, strategy_center, strategy_border)
    _, prob, _ = model_pvp(N, qr, num_steps=num_steps)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() > 0:
        prob /= prob.sum()
    return qr, probability_optimal, prob


def solve_pvp_geometric_center_sweep(
    epsilon_values,
    strategy_border_path,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=9999,
):
    strategy_border = np.load(strategy_border_path).astype(float)

    if strategy_border.shape[0] != strategy_border.shape[1]:
        raise ValueError("Border strategy must be a square matrix.")

    N = strategy_border.shape[0] - 1
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

    for epsilon in tqdm(epsilon_values, desc="Building geometric PvP center strategy"):
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
        "strategy_border": strategy_border,
        "mode": "geometric-center",
    }
