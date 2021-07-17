import random
import numpy as np
from tqdm import tqdm

from get_neighbours import get_neighbours

# This function represents sampling process.
# A point is placed in the middle of a square N by N 
# and it moves with given probabilities P from state to state
# until it reaches border. There are 10^6 games simulated here.
# N - index of the last item in line of a square. 
# P - transition matrix consisting of matrices Q and R. 
# Q describes the probability of transitioning from some transient state to another while
# R describes the probability of transitioning from some transient state to some absorbing state. 

def generate_distribution(N, P):
    states = []
    turns_arr = []
    for i in tqdm(range(0, 1000000)):
        up = 0
        right = 0
        turns = 0
        state = (8, 8)
        states.append(state)
        while (abs(up) < 8 and abs(right) < 8):
            x = random.uniform(0, 1)
            possible_next = get_neighbours(state)
            p1 = P[(N + 1) * state[0]+ state[1], (N + 1) * possible_next[0][0] + possible_next[0][1]]
            if (x >= 0 and x < p1):
                right -= 1
                turns += 1
                state = (possible_next[0][0], possible_next[0][1])
            if (len(possible_next) > 1):
                p2 = P[(N + 1) * state[0]+ state[1], (N + 1) * possible_next[1][0] + possible_next[1][1]]
            if (x >= p1 and x < p1 + p2):
                right += 1
                turns += 1
                state = (possible_next[1][0], possible_next[1][1])
            if (len(possible_next) > 2):
                p3 = P[(N + 1) * state[0]+ state[1], (N + 1) * possible_next[2][0] + possible_next[2][1]]
            if (x >= p1+p2 and x < p1+p2+p3):
                up += 1
                turns += 1
                state = (possible_next[2][0], possible_next[2][1])
            if (len(possible_next) > 3):
                p4 = P[(N + 1) * state[0]+ state[1], (N + 1) * possible_next[3][0] + possible_next[3][1]]
            if (x >= p1+p2+p3 and x < 1):
                up -= 1
                turns += 1
                state = (possible_next[3][0], possible_next[3][1])
            #states.append(state)
        turns_arr.append(turns)
    return turns_arr#, states