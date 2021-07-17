import numpy as np

from is_close import is_close

# Divides an array of states into separate games.
# states - array of states that contains at least 1 game.

def get_game_trajectories(states):
    games = []
    temp = []
    i = 0
    for i in np.arange(1, len(states)):
        if (is_close(states[i], states[i-1])):
            temp.append(states[i-1])
        else:
            temp.append(states[i-1])
            games.append(temp)
            temp = []
    temp.append(states[i])
    games.append(temp)
    return games