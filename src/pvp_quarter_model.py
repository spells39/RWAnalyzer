import numpy as np
from numpy.linalg import LinAlgError
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp
from bvp_quarter_model import build_border_strategy
from pvb_quarter_model import build_center_strategy
from quarter_partition_utils import ensure_output_dirs, find_mean_time_banded


def compute_duration_distribution_from_strategies(N, strategy_center, strategy_border, num_steps=9999):
    qr, probability_optimal = make_prob_matrix(N, 1.0 - strategy_center, 1.0 - strategy_border)
    _, prob, _ = model_pvp(N, qr, num_steps=num_steps)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() > 0:
        prob /= prob.sum()
    return qr, probability_optimal, prob


def solve_pvp_quarter_sweep(
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=9999,
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

    for epsilon in tqdm(epsilon_values, desc="Building quarter-based PvP strategies"):
        strategy_center = build_center_strategy(n, epsilon)
        strategy_border = build_border_strategy(n, epsilon)
        qr_optimal, probability_optimal, prob = compute_duration_distribution_from_strategies(
            N,
            strategy_center,
            strategy_border,
            num_steps=num_steps,
        )
        np.save(qr_matrices + f"qr_{epsilon:.2f}", qr_optimal)

        try:
            mean_time, state_mean_times = find_mean_time_banded(probability_optimal, N - 1)
        except LinAlgError:
            mean_time = float("inf")
            state_mean_times = np.full(inner_n ** 2, np.inf, dtype=float)
        mean_times.append(mean_time)

        fit_score = None
        if real_pmf is not None:
            fit_score = float(jensenshannon(real_pmf, prob)) if prob.sum() > 0 else float("inf")
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
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.array(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "mode": "quarter-combined-pvb-center-bvp-border",
    }
