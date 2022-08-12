def digitalize_states(series):
    """
    This function takes states as strings (as they are exported from xls)
    and converts them to pairs of integers.
    series - array of states, where each state is a string.
    """
    states = []
    for point in series:
        if isinstance(point, str):
            point = point[1:-1]
            point = point.split(",")
            x = int(point[0])
            y = int(point[1])
            pair = (x, y)
        else:
            pair = point
        states.append(pair)
    return states