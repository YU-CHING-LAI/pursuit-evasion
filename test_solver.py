import numpy as np
from env import PursuitEvasionGrid
from baseline import evaluate, random_policy
from solver import value_iteration, optimal_policies

# optimal pursuer & optimal evader
env_no = PursuitEvasionGrid(n=10, seed=0)
env_hole = PursuitEvasionGrid(n=10, obstacles=[(4, 4), (4, 5)], seed=0)
env_wall = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
env_set = [env_no, env_hole, env_wall]
names = ["no obstacle", "hole", "wall"]

for name, env in zip(names, env_set):
    print(f"\n====== optimal pursuer & optimal evader with {name} ======\n")
    V0, V1, iters = value_iteration(env)
    print("iteration:", iters)

    optimal_pursuer, optimal_evader = optimal_policies(V0, V1, env.neighbors)
    rate, steps = evaluate(env, optimal_pursuer, optimal_evader)
    print(f"capture rate: {rate:.1%}")
    print(f"average steps to capture: {steps:.1f}")

# optimal pursuer & random
env_random_evader = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
print(f"\n====== optimal pursuer & random evader with wall ======\n")
V0, V1, iters = value_iteration(env_random_evader)
print("iteration:", iters)
optimal_pursuer, optimal_evader = optimal_policies(V0, V1, env_random_evader.neighbors)
rate, steps = evaluate(env_random_evader, optimal_pursuer, random_policy)
print(f"capture rate: {rate:.1%}")
print(f"average steps to capture: {steps:.1f}")

# Adjacent, pursuer's turn: capture on the next move
p, e = env.to_idx(2, 7), env.to_idx(2, 8)
assert V0[p, e] == 1, f"Adjacent on pursuer's turn: expected V0 = 1, got {V0[p, e]}"
# Adjacent, evader's turn: the evader moves first, so at least 2 steps
assert V1[p, e] >= 2, f"Adjacent on evader's turn: expected V1 >= 2, got {V1[p, e]}"
# Evader trapped in a corner: exactly 2 steps
p, e = env.to_idx(1, 1), env.to_idx(0, 0)
assert V1[p, e] == 2, f"Evader in corner: expected V1 = 2, got {V1[p, e]}"