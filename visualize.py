import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env import PursuitEvasionGrid
from baseline import random_policy

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


def animate_episode(env, pursuer_policy, evader_policy, path):
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
    fig.suptitle("Random pursuer vs Random evader")

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

    env = PursuitEvasionGrid(n=10, obstacles=[(4, 4), (4, 5)], seed=0)
    path = "outputs/animate_test.gif"
    animate_episode(env, random_policy, random_policy, path)
    print("done, total steps: ", env.t)
    

    