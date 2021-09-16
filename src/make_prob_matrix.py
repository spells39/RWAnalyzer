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
    for x, y in product(range(1, N), range(1, N)):
            pc = strategy_center[x, y]
            pb = strategy_border[x, y]
            probabilities[(N+1) * x + y, (N+1) * (x + 0) + (y + 1)] = (1 - pc) * pb
            probabilities[(N+1) * x + y, (N+1) * (x + 0) + (y - 1)] = pc * pb
            probabilities[(N+1) * x + y, (N+1) * (x + 1) + (y + 0)] = pc * (1 - pb)
            probabilities[(N+1) * x + y, (N+1) * (x - 1) + (y + 0)] = (1 - pc) * (1 - pb)
    
    qr = probabilities.copy()
    
    border = get_border_cases(N)
    for i in np.arange(len(border)-1, -1, step = -1):
        probabilities = np.delete(probabilities, border[i], axis = 0)
        probabilities = np.delete(probabilities, border[i], axis = 1)
    return qr, probabilities