# This function generates a game that is symmetrical
# to the original game across the main diagonal of the square.
# N - index of the last item in line of a square. 
# game - array of states, to which a symmetrical one will be generated.

def get_main_diag_symmetry(N, game):
    symm_game = []
    for state in game:
        symm_game.append((N - state[1], N - state[0]))
    return symm_game