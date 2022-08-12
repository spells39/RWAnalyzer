import numpy as np
from numba import njit 
from numba.types import bool_

from digitalize_states import digitalize_states
from get_border_cases import get_border_cases
from is_close import is_close

@njit
def make_prob_matrix(N, strategy_center, strategy_border):
    """    
    Forms a probability matrix Q from given startegies.
    Q describes the probability of transitioning from some transient state to another.
    N - index of the last item in line of a square. 
    strategy_center - matrix of probabilities for the first pure center strategy
    strategy_border - matrix of probabilities for the first pure border strategy
    """
    qr = np.zeros(((N+1)**2, (N+1)**2))
    for x in range(1, N):
        for y in range(1, N):
            pc = strategy_center[x, y]
            pb = strategy_border[x, y]
            qr[(N+1) * x + y, (N+1) * (x + 0) + (y + 1)] = (1 - pc) * pb
            qr[(N+1) * x + y, (N+1) * (x + 0) + (y - 1)] = pc * pb
            qr[(N+1) * x + y, (N+1) * (x + 1) + (y + 0)] = pc * (1 - pb)
            qr[(N+1) * x + y, (N+1) * (x - 1) + (y + 0)] = (1 - pc) * (1 - pb)

    probabilities = np.zeros(((N-1)**2, (N-1)**2))
    for x in range(0, N - 1):
        for y in range(0, N - 1):
            pc = strategy_center[x + 1, y + 1]
            pb = strategy_border[x + 1, y + 1]
            if y + 1 < N - 1:
                probabilities[(N-1) * x + y, (N-1) * (x + 0) + (y + 1)] = (1 - pc) * pb
            if y - 1 >= 0:
                probabilities[(N-1) * x + y, (N-1) * (x + 0) + (y - 1)] = pc * pb
            if x + 1 < N - 1:
                probabilities[(N-1) * x + y, (N-1) * (x + 1) + (y + 0)] = pc * (1 - pb)
            if x - 1 >= 0:
                probabilities[(N-1) * x + y, (N-1) * (x - 1) + (y + 0)] = (1 - pc) * (1 - pb)
    
    return qr, probabilities

