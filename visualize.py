import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env import PursuitEvasionGrid
from baseline import random_policy
from solver import value_iteration, optimal_policies

def draw_frame(env, ax):
    ax.imshow(env.blocked, cmap="Greys")
    pr, pc = env.to_rc(env.pursuer) 
    er, ec = env.to_rc(env.evader)     
    ax.scatter(pc, pr, c="red", s=200, label="pursuer")
    ax.scatter(ec, er, c="blue", s=200, label="evader")
    ax.legend(loc="upper right")


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]


def animate_episode(env, pursuer_policy, evader_policy, path, title=""):
    state = env.reset()
    states = [(env.pursuer, env.evader, env.turn, env.t)]
    done = False
    while not done: 
        if env.turn == 0: 
            action = pursuer_policy(env, state)
        else: 
            action = evader_policy(env, state)
        state, reward, done = env.step(action)
        states.append((env.pursuer, env.evader, env.turn, env.t))
    fig, ax = plt.subplots()
    fig.suptitle(title)  

    def update(i):
        ax.clear()
        env.pursuer, env.evader, env.turn, env.t = states[i]   
        draw_frame(env, ax)
        if env.pursuer == env.evader:
            ax.set_title(f"t = {env.t}, caught!")
        else:
            ax.set_title(f"t = {env.t}")

    anim = FuncAnimation(fig, update, frames=len(states), interval=200)
    anim.save(path, writer="pillow")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    env_ran_ran = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
    path_ran = "outputs/random_vs_random.gif"
    animate_episode(env_ran_ran, random_policy, random_policy, path_ran, "random pursuer v.s. random evader")
    print("done, total steps: ", env_ran_ran.t)

    env_opt_ran = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
    V0, V1, iters = value_iteration(env_opt_ran)
    optimal_pursuer, optimal_evader = optimal_policies(V0, V1, env_opt_ran.neighbors)
    path_opt = "outputs/optimal_vs_random.gif"
    animate_episode(env_opt_ran, optimal_pursuer, random_policy, path_opt, "optimal pursuer v.s. random evader")
    print("done, total steps: ", env_opt_ran.t)

    env_opt_opt = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
    V0, V1, iters = value_iteration(env_opt_opt)
    optimal_pursuer, optimal_evader = optimal_policies(V0, V1, env_opt_opt.neighbors)
    path_opt = "outputs/optimal_vs_optimal.gif"
    animate_episode(env_opt_opt, optimal_pursuer, optimal_evader, path_opt, "optimal pursuer v.s. optimal evader")
    print("done, total steps: ", env_opt_opt.t)


    