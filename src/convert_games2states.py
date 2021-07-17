# This function takes game trajectories produced
# by get_game_trajectories() function and turns
# them into single array of states.
# games - array of games (each game array has 4 games that are symmetrical).

def convert_games2states(games):
    states = []
    for symm_games in games:
        for game in symm_games:
            for state in game:
                states.append(state)
    return states