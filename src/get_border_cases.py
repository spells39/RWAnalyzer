import numpy as np
from numba import njit

# This function counts border states for a square.
# N - index of last item in line (counting starts from 0).
# So if there are 17 states in line, N will be 16.

@njit
def get_border_cases(N):
    border_cases = []
    for i in [0, N]:
        for j in np.arange(0, N + 1):
            border_cases.append(i * (N + 1) + j)
            if i != j and i + j != N:
                border_cases.append(j * (N + 1) + i)
    return border_cases