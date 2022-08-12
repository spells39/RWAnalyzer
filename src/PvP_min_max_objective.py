import numpy as np
from numba import njit

def get_mean_time(M, fundamental_matrix):
    c = np.ones(fundamental_matrix.shape[0])
    mean_times = fundamental_matrix @ c
    return mean_times[M * (M // 2) + M // 2]

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

def objective_function(strategy_center, strategy_border, M):
    probability_matrix = make_prob_matrix(M, strategy_center, strategy_border)
    fundamental_matrix = get_fundamental_matrix(probability_matrix)
    mean_time = get_mean_time(M, fundamental_matrix)
    return mean_time

'''
Problem:
max by strategy_center
    min by strategy_border
        objective_function(strategy_center, strategy_border)
N = const = 16 - the size of problem
'''

N = 16
M = N - 1

strategy_center = np.ones ((M, M), dtype=np.float64) * 0.5
strategy_border = np.ones ((M, M), dtype=np.float64) * 0.5

# 75.20846497681381
print(objective_function(strategy_center, strategy_border, M))