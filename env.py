import numpy as np

moves = []

for dr in (-1, 0, 1): # delta row
    for dc in (-1, 0, 1): # delta column
        moves.append((dr, dc))

class PursuitEvasionGrid:
    def __init__(self, n=10, obstacles=None, max_steps=100, seed=None): # n is the side length of grid graph
        self.n, self.max_steps = n, max_steps
        self.rng = np.random.default_rng(seed)
        self.blocked = np.zeros((n, n), dtype=bool) # whether the grid is occupied by obstacle
        for r, c in (obstacles or []): # mark obstacles
            self.blocked[r, c] = True
        self.neighbors = self._build_neighbors()
        
    def to_idx(self, r, c): # transfer into index
        return r * self.n + c

    def to_rc(self, idx): # transfer into row and column 
        return divmod(idx, self.n)

    def _build_neighbors(self):
        table = np.zeros((self.n*self.n, len(moves)), dtype=int)
        for idx in range(self.n*self.n):
            r, c = self.to_rc(idx)
            for a, (dr, dc) in enumerate(moves):
                if 0 <= r+dr < self.n and 0 <= c+dc < self.n and not self.blocked[r+dr, c+dc]:
                    table[idx, a] = self.to_idx(r+dr, c+dc)
                else:
                    table[idx, a] = idx
        return table
        
    def reset(self):
        self.t = 0 # time steps     
        self.turn = 0 # turn 0 is pursuer, 1 is evader
        free = np.nonzero(~self.blocked.flatten())[0]
        p, e = self.rng.choice(free, size=2, replace=False) # choose two grids that are not occupied
        self.pursuer, self.evader = int(p), int(e)
        return self._state()

    def step(self, action):
        if self.turn == 0:
            self.pursuer = int(self.neighbors[self.pursuer, action])
        else:
            self.evader = int(self.neighbors[self.evader, action])
        
        self.t += 1 # increase when either pursuer or evader moves
        
        caught = self.pursuer == self.evader
        reward = 1 if caught else 0
        done = caught or self.t >= self.max_steps

        self.turn = 1 - self.turn

        return self._state(), reward, done
            

    def _state(self):
        return (self.pursuer, self.evader, self.turn)