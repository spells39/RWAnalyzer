import numpy as np

from digitalize_states import digitalize_states
from get_border_cases import get_border_cases
from is_close import is_close

def make_prob_matrix(N, strategy_center, strategy_border):
    """    
    Forms a probability matrix Q from given startegies.
    Q describes the probability of transitioning from some transient state to another.
    N - index of the last item in line of a square. 
    strategy_center - matrix of probabilities for the first pure center strategy
    strategy_border - matrix of probabilities for the first pure border strategy
    """
    probabilities = np.zeros(((N+1)**2, (N+1)**2))
    from itertools import product
    for i, j in product(range(1, N), range(1, N)):
            pc = strategy_center[i, j]
            pb = strategy_border[i, j]
            probabilities[(N+1) * i + j, (N+1) * (i + 0) + (j + 1)] = (1 - pc) * pb
            probabilities[(N+1) * i + j, (N+1) * (i + 0) + (j - 1)] = pc * pb
            probabilities[(N+1) * i + j, (N+1) * (i + 1) + (j + 0)] = pc * (1 - pb)
            probabilities[(N+1) * i + j, (N+1) * (i - 1) + (j + 0)] = (1 - pc) * (1 - pb)
    
    qr = probabilities.copy()
    
    border = get_border_cases(N)
    for i in np.arange(len(border)-1, -1, step = -1):
        probabilities = np.delete(probabilities, border[i], axis = 0)
        probabilities = np.delete(probabilities, border[i], axis = 1)
    return qr, probabilities