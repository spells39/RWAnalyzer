import numpy as np

# Calculates the expected number of steps before
# the chain is absorbed at some absorbing state
# when starting in the center of the square
# N - index of the last item in line of a square. 
# fundamental_matrix - fundamental matrix of this Markov chain.

def get_mean_time(N, fundamental_matrix):
    c = np.ones(fundamental_matrix.shape[0])
    mean_times = fundamental_matrix @ c
    return mean_times[N * (N // 2) + N // 2]