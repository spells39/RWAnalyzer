import numpy as np


def model_pvp(N, P, num_steps = 10000):
    """
    Models a PvP game via probability propagation.
    N - index of the last item in line of a square. 
    P - transition matrix consisting of matrices Q and R. 
    Q describes the probability of transitioning from some transient state to another while
    R describes the probability of transitioning from some transient state to some absorbing state.  
    """
    a = np.zeros((N + 1, N + 1), dtype = np.int64).tolist()
    a[N // 2][N // 2] = 1
    b = []
    
    for k in range(num_steps + 1):
        b += [a]
        c = [[0] * (N + 1) for i in range(N + 1)]
        for i in range(1, N):
            for j in range(1, N):
                if i < N:
                    c[i + 1][j] += P[(N + 1) * i + j, (N + 1) * (i + 1) + (j + 0)] * a[i][j]
                if j < N:             
                    c[i][j + 1] += P[(N + 1) * i + j, (N + 1) * (i + 0) + (j + 1)] * a[i][j]
                if i > 0:
                    c[i - 1][j] += P[(N + 1) * i + j, (N + 1) * (i - 1) + (j - 0)] * a[i][j]
                if j > 0:
                    c[i][j - 1] += P[(N + 1) * i + j, (N + 1) * (i - 0) + (j - 1)] * a[i][j]           
        a = c
    d = []
    prob = []
    for k in range(num_steps + 1):
        res = 0
        for i in range(N + 1):
            res += b[k][i][0] + b[k][0][i] + b[k][-1][i] + b[k][i][-1]    
        d += [(k - 0) * res]
        
    even = 0
    odd = 0
    b = np.array(b)
    for k in range (num_steps + 1):
        val = b[k, :, 0] + b[k, :, -1] + b[k, 0, :] + b[k, -1, :]
        if (k % 2 == 0):
            even += sum(val)
        else:
            odd += sum(val)
        prob.append(sum(val))
    return d, prob, (even, odd)