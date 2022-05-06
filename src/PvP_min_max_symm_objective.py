import numpy as np
from numba import njit

def get_mean_time(fundamental_matrix, cid):
    c = np.ones(fundamental_matrix.shape[0])
    mean_times = fundamental_matrix @ c
    return mean_times[cid]

def get_fundamental_matrix(Q):
    return np.linalg.inv(np.eye(Q.shape[0]) - Q)

@njit
def make_prob_matrix(M, strategy_center, strategy_border):
    probabilities = np.zeros((M ** 2, M ** 2))
    for x in range(0, M):
        for y in range(0, M):
            pc = strategy_center[x, y]
            pb = strategy_border[x, y]
            if y + 1 < M:
                probabilities[M * x + y, M * (x + 0) + (y + 1)] = (1 - pc) * pb
            if y - 1 >= 0:
                probabilities[M * x + y, M * (x + 0) + (y - 1)] = pc * pb
            if x + 1 < M:
                probabilities[M * x + y, M * (x + 1) + (y + 0)] = pc * (1 - pb)
            if x - 1 >= 0:
                probabilities[M * x + y, M * (x - 1) + (y + 0)] = (1 - pc) * (1 - pb)
    
    return probabilities

@njit
def make_prob_matrix_symm(M, strategy_center, strategy_border):
    probabilities = np.zeros((M ** 2, M ** 2))
    for x in range(0, M):
        for y in range(0, M):
            if x + y == M - 1 and x == y:
                probabilities[M * x + y, M * (x + 0) + (y - 1)] = 1
            elif x + y < M and x - y >= 0:
                pc = strategy_center[x, y]
                pb = strategy_border[x, y]
                if y + 1 < M:
                    if x == y:
                        probabilities[M * x + y, M * (x + 1) + (y + 0)] += (1 - pc) * pb
                    elif x + y == M - 1:
                        probabilities[M * x + y, M * (x - 1) + (y + 0)] += (1 - pc) * pb
                    else:
                        probabilities[M * x + y, M * (x + 0) + (y + 1)] += (1 - pc) * pb
                if y - 1 >= 0:
                    probabilities[M * x + y, M * (x + 0) + (y - 1)] += pc * pb
                if x + 1 < M:
                    if x + y == M - 1:
                        probabilities[M * x + y, M * (x + 0) + (y - 1)] += pc * (1 - pb)
                    else:
                        probabilities[M * x + y, M * (x + 1) + (y + 0)] += pc * (1 - pb)
                if x - 1 >= 0:
                    if x == y:
                        probabilities[M * x + y, M * (x + 0) + (y - 1)] += (1 - pc) * (1 - pb)
                    else:
                        probabilities[M * x + y, M * (x - 1) + (y + 0)] += (1 - pc) * (1 - pb)
    
    return probabilities

def delete_zero_rows(probabilities, cid):
    ids = np.flatnonzero(np.all(probabilities == 0, 1) & np.all(probabilities == 0, 0))
    probabilities = np.delete(probabilities, ids, 0)
    probabilities = np.delete(probabilities, ids, 1)
    return probabilities, cid - (ids < cid).sum()

@njit
def iter_triangle(M):
    for x in range(M):
        for y in range(M):
            if x + y < M - 1 and x - y > 0:
                yield x, y

@njit
def prepare_matrix(strategy_center_flat, strategy_border_flat, M):
    strategy_center = np.zeros ((M, M), dtype=dtype)
    strategy_border = np.zeros ((M, M), dtype=dtype)

    for k, (x, y) in enumerate(iter_triangle(M)):
        p = strategy_center_flat[k]
        strategy_center[x, y] = p
        strategy_center[y, x] = 1 - p
        strategy_center[M - y - 1, M - x - 1] = p
        strategy_center[M - x - 1, M - y - 1] = 1 - p
    for i in range(M):
        if i * 2 < M - 1:
            strategy_center[M - i - 1, i] = 0
        elif i * 2 > M:
            strategy_center[M - i - 1, i] = 1
        else:
            strategy_center[M - i - 1, i] = 0.5
        strategy_center[i, i] = 0.5
    #strategy_center += np.fliplr(np.diag(np.full(strategy_center.shape[0], 0.5)))
    #strategy_center += np.diag(np.full(strategy_center.shape[0], 0.5)) - np.diag(np.diag(strategy_center))

    for k, (x, y) in enumerate(iter_triangle(M)):
        p = strategy_border_flat[k]
        strategy_border[x, y] = p
        strategy_border[y, x] = 1 - p
        strategy_border[M - y - 1, M - x - 1] = 1 - p
        strategy_border[M - x - 1, M - y - 1] = p
    strategy_border += np.fliplr(np.diag(np.full(strategy_border.shape[0], 0.5)))
    strategy_border += np.diag(np.full(strategy_border.shape[0], 0.5)) - np.diag(np.diag(strategy_border))
    return strategy_center, strategy_border

def objective_function(strategy_center_flat, strategy_border_flat, M):
    strategy_center, strategy_border = prepare_matrix(strategy_center_flat, strategy_border_flat, M)
    probability_matrix_symm = make_prob_matrix_symm(M, strategy_center, strategy_border)
    cid = M * (M // 2) + M // 2
    probability_matrix_symm, cid = delete_zero_rows(probability_matrix_symm, cid)
    fundamental_matrix_symm = get_fundamental_matrix(probability_matrix_symm)
    mean_time_symm = get_mean_time(fundamental_matrix_symm, cid)
    return mean_time_symm
'''
Problem:
max by strategy_center
    min by strategy_border
        objective_function(strategy_center_flat, strategy_border_flat)
N = const = 16 - the size of problem
cnt = const = 49 - the number of variables in the first vector and the second vector
Overall 49 * 2 = 98 variables
Expected answer is around 146, more broadly in the range [64, 225]
'''

N = 16
M = N - 1
dtype = np.float64
cnt = (M ** 2 - 1) // 4 - M // 2

# example of very high value on the input that close to the boundary of variable space
# 12106879.998128386
strategy_center_flat = np.ones(cnt, dtype=dtype) * 0.1
strategy_border_flat = np.ones(cnt, dtype=dtype) * 1

# 32.151874764517075
#strategy_center_flat = np.linspace(0, 1, cnt)
#strategy_border_flat = np.linspace(0, 1, cnt)

# random
#strategy_center_flat = np.random.rand(cnt)
#strategy_border_flat = np.random.rand(cnt)

print(objective_function(strategy_center_flat, strategy_border_flat, M))

# 49
print(cnt)