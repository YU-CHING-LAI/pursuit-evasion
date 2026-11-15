import numpy as np
from env import PursuitEvasionGrid

def value_iteration(env, max_iter=1000): # Minimax Value Iteration
    N = env.neighbors
    n_square = env.n**2

    V0 = np.full((n_square, n_square), np.inf) # (p, e)
    V1 = np.full((n_square, n_square), np.inf)
    np.fill_diagonal(V0, 0)
    np.fill_diagonal(V1, 0)

    for it in range(max_iter):
        V0_new = 1 + V1[N].min(axis=1) # Bellman Equation
        V1_new = 1 + V0[:, N].max(axis=2) 
        # V1[N][p, a, e] is (n^2, 9, n^2), .min(axis=1) choose the min over 9 actions
        # V0[:, N][p, e, a] is (n^2, n^2, 9), .max(axis=2) choose the max over 9 actions
        # V0_new and V1_new are (n^2, n^2)
        np.fill_diagonal(V0_new, 0)
        np.fill_diagonal(V1_new, 0)
        if np.array_equal(V0, V0_new) and np.array_equal(V1, V1_new):
            break
        V0, V1 = V0_new, V1_new

    return V0, V1, it+1 # it+1 iteration

def optimal_policies(V0, V1, N):
    def optimal_pursuer(env, state):
        p, e, turn = state
        return int(np.argmin(V1[N[p], e]))
    def optimal_evader(env, state):
        p, e, turn = state
        return  int(np.argmax(V0[p, N[e]]))
    return optimal_pursuer, optimal_evader
