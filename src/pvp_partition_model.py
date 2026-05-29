import os
import tempfile

import numpy as np
import scipy.linalg
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from bvp_partition_model import solve_bvp_sweep_legacy
from make_prob_matrix import make_prob_matrix
from model_pvp import model_pvp
from pvb_partition_model import solve_pvb_sweep


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


def _component_paths(output_duration, component_name):
    root = os.path.join(output_duration, "components", component_name)
    return {
        "output_duration": root + os.sep,
        "output_absorption_images1": os.path.join(root, "absorption_times") + os.sep,
        "output_absorption_images2": os.path.join(root, "center_strategies") + os.sep,
        "output_absorption_images3": os.path.join(root, "border_strategies") + os.sep,
        "qr_matrices": os.path.join(root, "qr") + os.sep,
    }


def _build_combined_transition_matrix(N, inner_n, center_snapshot, border_snapshot):
    center_up_right = np.reshape(center_snapshot["p1s"], (inner_n, inner_n))
    border_vertical = np.reshape(border_snapshot["q1s"], (inner_n, inner_n))

    strategy_center_for_matrix = np.pad(
        1.0 - center_up_right,
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    strategy_border_for_matrix = np.pad(
        border_vertical,
        pad_width=1,
        mode="constant",
        constant_values=0,
    ).T
    return make_prob_matrix(N, strategy_center_for_matrix, strategy_border_for_matrix)


def compute_pvp_movement_diagnostics(qr, N):
    inner_n = N - 1
    expected_col_step = np.zeros((inner_n, inner_n), dtype=float)
    expected_row_step = np.zeros((inner_n, inner_n), dtype=float)
    border_progress = np.zeros((inner_n, inner_n), dtype=float)

    for row in range(1, N):
        for col in range(1, N):
            source_index = (N + 1) * row + col
            probabilities = {
                (-1, 0): qr[source_index, (N + 1) * (row - 1) + col],
                (1, 0): qr[source_index, (N + 1) * (row + 1) + col],
                (0, -1): qr[source_index, (N + 1) * row + col - 1],
                (0, 1): qr[source_index, (N + 1) * row + col + 1],
            }
            target_row = row - 1
            target_col = col - 1
            current_border_distance = min(row, col, N - row, N - col)
            expected_distance = 0.0
            for (row_step, col_step), probability in probabilities.items():
                expected_row_step[target_row, target_col] += probability * row_step
                expected_col_step[target_row, target_col] += probability * col_step
                next_row = row + row_step
                next_col = col + col_step
                expected_distance += probability * min(
                    next_row,
                    next_col,
                    N - next_row,
                    N - next_col,
                )
            border_progress[target_row, target_col] = current_border_distance - expected_distance

    transient_indices = [
        (N + 1) * row + col
        for row in range(1, N)
        for col in range(1, N)
    ]
    transient_matrix = qr[np.ix_(transient_indices, transient_indices)]
    initial_state = np.zeros(inner_n ** 2, dtype=float)
    initial_state[(inner_n // 2) * inner_n + inner_n // 2] = 1.0
    expected_visits = scipy.linalg.solve(
        np.eye(transient_matrix.shape[0]) - transient_matrix.T,
        initial_state,
    ).reshape((inner_n, inner_n))
    visit_share = expected_visits / expected_visits.sum()

    return {
        "expected_col_step": expected_col_step,
        "expected_row_step": expected_row_step,
        "border_progress": border_progress,
        "expected_visits": expected_visits,
        "visit_share": visit_share,
    }


def sample_pvp_trajectory(qr, N, max_steps=10000, random_seed=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")

    rng = np.random.default_rng(random_seed)
    current_row = N // 2
    current_col = N // 2
    trajectory = [(current_row, current_col)]

    for _ in range(max_steps):
        if current_row in (0, N) or current_col in (0, N):
            return np.asarray(trajectory, dtype=int), True

        state_index = (N + 1) * current_row + current_col
        probabilities = np.asarray(qr[state_index], dtype=float)
        total_probability = probabilities.sum()
        if not np.isclose(total_probability, 1.0):
            if total_probability <= 0.0:
                raise RuntimeError("No outgoing transition is available for the current state.")
            probabilities = probabilities / total_probability

        next_index = int(rng.choice(qr.shape[1], p=probabilities))
        current_row, current_col = divmod(next_index, N + 1)
        trajectory.append((current_row, current_col))

    absorbed = current_row in (0, N) or current_col in (0, N)
    return np.asarray(trajectory, dtype=int), absorbed


def solve_pvp_combined_sweep(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    real_pmf_path=None,
    num_steps=999,
    max_attempts=100,
    save_component_outputs=False,
):
    N = n - 1
    inner_n = n - 2
    epsilon_values = np.asarray(epsilon_values, dtype=float)
    if np.any(epsilon_values <= 0.0):
        raise ValueError(
            "Combined PvP requires epsilon > 0: at epsilon=0 the two pure "
            "baseline policies can form a non-absorbing cycle."
        )
    ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices)

    if save_component_outputs:
        component_workspace = output_duration
        cleanup_workspace = None
    else:
        cleanup_workspace = tempfile.TemporaryDirectory(prefix="_pvp_components_", dir=output_duration)
        component_workspace = cleanup_workspace.name + os.sep
    try:
        center_result = solve_pvb_sweep(
            n=n,
            epsilon_values=epsilon_values,
            real_pmf_path=None,
            num_steps=num_steps,
            max_attempts=max_attempts,
            **_component_paths(component_workspace, "pvb_center"),
        )
        border_result = solve_bvp_sweep_legacy(
            n=n,
            epsilon_values=epsilon_values,
            max_attempts=max_attempts,
            **_component_paths(component_workspace, "bvp_border"),
        )
    finally:
        if cleanup_workspace is not None:
            cleanup_workspace.cleanup()

    real_pmf = None
    if real_pmf_path is not None:
        real_pmf = np.load(real_pmf_path)[: num_steps + 1].astype(float)
        if real_pmf.sum() > 0:
            real_pmf /= real_pmf.sum()

    with open(output_duration + "epsilon_values.txt", "w") as file:
        for value in epsilon_values:
            file.write(f"{value:.3f}\n")

    strategy_snapshots = []
    mean_times = []
    fit_scores = []
    for epsilon, center_snapshot, border_snapshot in tqdm(
        zip(epsilon_values, center_result["strategy_snapshots"], border_result["strategy_snapshots"]),
        total=len(epsilon_values),
        desc="Combining PvB and BvP policies",
    ):
        qr_optimal, probability_optimal = _build_combined_transition_matrix(
            N,
            inner_n,
            center_snapshot,
            border_snapshot,
        )
        mean_time, state_mean_times = find_mean_time_banded(probability_optimal, N - 1)
        movement_diagnostics = compute_pvp_movement_diagnostics(qr_optimal, N)
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
                "p1s": center_snapshot["p1s"],
                "q1s": border_snapshot["q1s"],
                "vs": state_mean_times,
                "w": None,
                "mean_time": float(mean_time),
                "jsd": fit_score,
                **movement_diagnostics,
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
        "epsilon_values": epsilon_values,
        "w_new_list": [],
        "strategy_snapshots": strategy_snapshots,
        "mean_times": np.asarray(mean_times, dtype=float),
        "fit_scores": fit_scores,
        "center_result": center_result,
        "border_result": border_result,
        "mode": "combined-pvb-center-vs-bvp-border",
        "component_outputs_saved": bool(save_component_outputs),
    }
