import numpy as np
from make_prob_matrix import make_prob_matrix

def traverse_paths(x, y, val, N, strategy, strategies_answers, num_strategies):
    if x < y:
        return
    if x == N - 1 and y == N - 1:
        prv = strategy[x][y]
        strategy[x][y] = val
        strategies_answers.append(strategy.copy())
        strategy[x][y] = prv
        return
    if x >= N or y >= N:
        return
    if len(strategies_answers) > num_strategies:
        return 
    prv = strategy[x][y]
    strategy[x][y] = val
    if np.random.randint(2):
        traverse_paths(x + 1, y, val, N, strategy, strategies_answers, num_strategies)
        traverse_paths(x, y + 1, val, N, strategy, strategies_answers, num_strategies)
    else:
        traverse_paths(x, y + 1, val, N, strategy, strategies_answers, num_strategies)
        traverse_paths(x + 1, y, val, N, strategy, strategies_answers, num_strategies)

    strategy[x][y] = prv



def make_banded_matrix(A, N):
    banded_matrix = np.zeros((2 * N + 1, N ** 2), dtype=np.float32)
    for i in [-N, -1, 0, 1, N]:
        d = np.diagonal(A, -i)
        if i < 0:
            banded_matrix[i + N, -i:] = d
        if i > 0:
            banded_matrix[i + N, :-i] = d
        if i == 0:
            banded_matrix[i + N, :] = d
    return banded_matrix

def find_mean_time_banded(A, N):
    banded_matrix = make_banded_matrix(np.eye(A.shape[0]) - A, N)
    import scipy
    mean_times = scipy.linalg.solve_banded((N, N), banded_matrix, np.ones(banded_matrix.shape[1]))
    return mean_times[N * (N // 2) + N // 2]

def find_mean_time(Q, N): # Sparse is slow
    mean_times = np.linalg.solve(np.eye(Q.shape[0]) - Q, np.ones(Q.shape[0]))
    return mean_times[N * (N // 2) + N // 2]


N = 16
num_strategies = 10000000

strategies_answers = []
strategy = np.full((N + 1, N + 1), 0, dtype=np.float64)
traverse_paths(x=1, y=1, val=1, N=N, strategy=strategy, strategies_answers=strategies_answers, num_strategies=num_strategies)

from tqdm import tqdm
N = 16
n = len(strategies_answers)
strategy_center = np.full((N + 1, N + 1), 0.5, dtype=np.float64)
strategy_border = np.full_like(strategy_center, 0.5)
cnt = 0
for i in (range(n)):
    for j in tqdm(range(n)):
        strategy_center[:, :] = 0.5
        strategy_center[strategies_answers[i].astype('bool').T] = 1
        strategy_center[strategies_answers[j].astype('bool')] = 0
        qr_optimal, probability_optimal = make_prob_matrix(N, strategy_center, strategy_border)
        
        mean_time = find_mean_time_banded(probability_optimal, N - 1)
        condition = np.abs(mean_time - 225) > 1e-6
        cnt += condition
        if condition:
            print(i, j, mean_time)
