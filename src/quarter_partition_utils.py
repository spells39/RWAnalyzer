import os

import numpy as np
import scipy.linalg


PURE_CENTER_UR = ("up", "right")
PURE_CENTER_DL = ("down", "left")
PURE_BORDER_VERTICAL = ("up", "down")
PURE_BORDER_HORIZONTAL = ("right", "left")

OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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
    return min(row, global_n - 1 - row, col, global_n - 1 - col)


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


def preferred_center_directions(global_index, global_n):
    quarter = get_quarter(global_index, global_n)
    return {
        0: set(),
        1: {"left", "down"},
        2: {"down"},
        3: {"right", "down"},
        4: {"right"},
        5: {"right", "up"},
        6: {"up"},
        7: {"left", "up"},
        8: {"left"},
    }[quarter]


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


def ensure_output_dirs(output_absorption_images1, output_absorption_images2, output_absorption_images3, qr_matrices):
    os.makedirs(output_absorption_images1, exist_ok=True)
    os.makedirs(output_absorption_images2, exist_ok=True)
    os.makedirs(output_absorption_images3, exist_ok=True)
    os.makedirs(qr_matrices, exist_ok=True)


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
