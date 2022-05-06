import numpy as np

from digitalize_states import digitalize_states
from get_border_cases import get_border_cases
from is_close import is_close

def get_games_hist_2d(N, games_states, return_counts=False):
    """
    Forms a 2D histogram matrix H from given game states.
    H describes the probability of piece being found in the particular state.
    N - index of the last item in line of a square. 
    games_states - array of states.
    """

    states = digitalize_states(games_states)
    counts = np.zeros((N+1, N+1), dtype=np.int64)
    for state in states:
        counts[state[0], state[1]] += 1
    frequencies = counts.astype(np.float64)
    hist_2d_inner = frequencies[1:N, 1:N]
    hist_2d_inner /= np.sum(hist_2d_inner)
    hist_2d_border = frequencies.copy()
    hist_2d_border[1:N, 1:N] = 0
    hist_2d_border /= np.sum(hist_2d_border)
    
    if return_counts:
        return hist_2d_inner, hist_2d_border, counts
    else:
        return hist_2d_inner, hist_2d_border