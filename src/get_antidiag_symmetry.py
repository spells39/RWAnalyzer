# This function generates a game that is symmetrical
# to the original game across the antidiagonal of the square.
# game - array of states, to which a symmetrical one will be generated.

def get_antidiag_symmetry(game):
    symm_game = []
    for state in game:
        symm_game.append((state[1], state[0]))
    return symm_game