import numpy as np

from digitalize_states import digitalize_states
from get_border_cases import get_border_cases
from is_close import is_close

# Forms a probability matrix Q from given game states.
# Q describes the probability of transitioning from some transient state to another.
# N - index of the last item in line of a square. 
# games_states - array of states.

def get_prob_matrix(N, games_states):
    states = digitalize_states(games_states)   
    frequencies = np.zeros(((N+1)**2, (N+1)**2))
    for i in range(1, len(states)):
        if (is_close(states[i], states[i-1])):
            frequencies[(N+1) * states[i-1][0] + states[i-1][1], 
                        (N+1) * states[i][0] + states[i][1]] +=1
    
    for i in range (0, frequencies.shape[0]):
        if (sum(frequencies[i]) != 0):
            frequencies[i] /= sum(frequencies[i])
    qr = frequencies
    
    border = get_border_cases(N)
    for i in np.arange(len(border)-1, -1, step = -1):
        frequencies = np.delete(frequencies, border[i], axis = 0)
        frequencies = np.delete(frequencies, border[i], axis = 1)
    return qr, frequencies