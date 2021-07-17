
# This function takes states as strings (as they are exported from xls)
# and converts them to pairs of integers.
# series - array of states, where each state is a string.

def digitalize_states(series):
    states = []
    for point in series:
        point = point[1:-1]
        point = point.split(",")
        x = int(point[0])
        y = int(point[1])
        pair = (x, y)
        states.append(pair)
    return states