import numpy as np

from digitalize_states import digitalize_states
from is_close import is_close

def get_games_strategy(N, games_states):
    """
    Forms a probability matrix Q from given game states.
    Q describes the probability of transitioning from some transient state to another.
    N - index of the last item in line of a square. 
    games_states - array of states.
    """
    states = digitalize_states(games_states)
    states = np.array(states)
    strategy_center = np.zeros((N+1, N+1), dtype=np.float64)
    strategy_border = np.zeros((N+1, N+1), dtype=np.float64)
    strategy_count = np.zeros((N+1, N+1), dtype=np.float64)
    for i in range(1, len(states)):
        if is_close(states[i], states[i-1]):
            x, y = states[i-1]
            dx, dy = states[i] - states[i-1]
            if dx == 0:
                strategy_border[x, y] += 1
            if dx == +1 or dy == -1:
                strategy_center[x, y] += 1
            strategy_count[x, y] += 1
    strategy_center /= np.maximum(1, strategy_count)
    strategy_border /= np.maximum(1, strategy_count)
    return strategy_border, strategy_center