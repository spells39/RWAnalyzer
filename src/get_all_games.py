from get_symmetrical_games import get_symmetrical_games

# This function takes an array of original games
# and for each game there it acquires symmetrical games.
# N - index of the last item in line of a square. 
# games - array of original games, to which symmetrical games will be generated.
# Returned array is 4 times larger than the original games array.

def get_all_games(N, games):
    symmetrical_games = []
    for game in games:
        symmetrical_games.append(get_symmetrical_games(N, game))
    return symmetrical_games