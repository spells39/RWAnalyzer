import numpy as np

from digitalize_states import digitalize_states
from is_close import is_close
from get_border_cases import get_border_cases
from get_all_games import get_all_games
from get_game_trajectories import get_game_trajectories
from convert_games2states import convert_games2states

# Forms a probability matrix Q from given game states while
# taking advantage of symmetrical properties of the square.
# Q describes the probability of transitioning from some transient state to another.
# N - index of the last item in line of a square. 
# games_states - array of states.

def get_prob_matrix_symmetry(N, games_states):
    states = digitalize_states(games_states)   
    frequencies = np.zeros(((N+1)**2, (N+1)**2))
    states = digitalize_states(games_states)
    states_symm = convert_games2states(get_all_games(N, get_game_trajectories(states)))
    
    for i in range(1, len(states_symm)):
        if (is_close(states_symm[i], states_symm[i-1])):
            frequencies[(N+1) * states_symm[i-1][0] + states_symm[i-1][1], 
                        (N+1) * states_symm[i][0] + states_symm[i][1]] +=1
    for i in range (0, frequencies.shape[0]):
        if (sum(frequencies[i]) != 0):
            frequencies[i] /= sum(frequencies[i])
    qr = frequencies.copy()
    border = sorted(get_border_cases(N)) # from greatest to smallest
    for i in np.arange(len(border)-1, -1, step = -1):
        frequencies = np.delete(frequencies, border[i], axis = 0)
        frequencies = np.delete(frequencies, border[i], axis = 1)
    return qr, frequencies