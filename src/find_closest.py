def find_closest(turns, games_turns):
    turns = turns // 1
    closest_index = -1
    k = 0
    while (closest_index == -1):
        if turns in games_turns:
            closest_index = games_turns.index(turns)
        else:
            turns += ((-1) ** k) * k
        k += 1
    return closest_index