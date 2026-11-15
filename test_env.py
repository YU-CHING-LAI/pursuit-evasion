from env import PursuitEvasionGrid

env = PursuitEvasionGrid(n=10, obstacles=[(2, 2)], seed=0)

# test moving and turn
print("\n====== moves and turns testing ======\n")
print("neighbor table size: ", env.neighbors.shape)

env.reset()
env.pursuer, env.evader, env.turn = env.to_idx(0, 9), env.to_idx(1, 2), 0
print("pursuer: ", env.pursuer) 
print("evader: ", env.evader)
print(env.step(2)) # out of boader
print(env.step(7)) # meet obstacle

# test catching
print("\n====== catch testing ======\n")
env.reset()
env.pursuer, env.evader, env.turn = env.to_idx(0, 3), env.to_idx(0, 4), 0
print("pursuer: ", env.pursuer) 
print("evader: ", env.evader)
print(env.step(5))

# timeout test 
print("\n====== timeout testing ======\n")
env.reset()
print("pursuer: ", env.pursuer) 
print("evader: ", env.evader)
env.t = env.max_steps-2
print(env.step(4))
env.t = env.max_steps-1
print(env.step(4))

# reset test
print("\n====== reset testing ======\n")
same = False
for i in range(0, 1000):
    env.reset()
    if env.pursuer == env.evader:
        same = True
if same:
    print("after reset, pursuer and evader have been on the same grid")
else:
    print("after reset, pursuer and evader have not been on the same grid")