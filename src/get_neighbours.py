# Acquires states that are reachable from this state
# within one move.
# state - current state, neighbours for which will be acquired.

def get_neighbours(state):
    neighbours = [(state[0] - 1, state[1]),
                  (state[0] + 1, state[1]),
                  (state[0], state[1] + 1),
                  (state[0], state[1] - 1)]
    return neighbours