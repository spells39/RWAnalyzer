# Checks if the two states are within one move from each other.
# state1 - first state.
# state2 - second state.

def is_close(state1, state2):
    (x1, y1) = (state1[0], state1[1])
    (x2, y2) = state2
    if (abs(x1 - x2) + abs(y1 - y2) == 1):
        return True
    else:
        return False