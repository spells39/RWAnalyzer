import numpy as np

# This function counts border states for a square.
# N - index of last item in line (counting starts from 0).
# So if there are 17 states in line, N will be 16.

def get_border_cases(N):
    border_cases = []
    for i in np.arange(0, N + 1):
        for j in np.arange(0, N + 1):
            if (i == 0 or i == N or j == 0 or j == N):
                border_cases.append(i * (N + 1) + j)
    return border_cases