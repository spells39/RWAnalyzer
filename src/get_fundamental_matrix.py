import numpy as np

# Forms a fundamental matrix for this Markov chain.
# Q - matrix that contains probability of 
#transitioning from some transient state to another.
def get_fundamental_matrix(Q):
    return np.linalg.inv(np.eye(Q.shape[0]) - Q)