from get_main_diag_symmetry import get_main_diag_symmetry
from get_antidiag_symmetry import get_antidiag_symmetry

# Generates symmetrical games for a given game across
# the main and the antidiagonal of the square.
# N - index of the last item in line of a square.
# game - game, for which the symmetrical games will be generated.

def get_symmetrical_games(N, game):
    symm_games = []
    symm_games.append(game)
    symm_games.append(get_main_diag_symmetry(N, game))
    symm_games.append(get_antidiag_symmetry(game))
    symm_games.append(get_main_diag_symmetry(N, symm_games[2]))
    return symm_games