from env import PursuitEvasionGrid, moves

def random_policy(env, state):
    return env.rng.choice(len(moves))


def run_episode(env, pursuer_policy, evader_policy):
    state = env.reset()
    done = False
    while not done: # until is done
        if env.turn == 0: # turn is changed for each time step
            action = pursuer_policy(env, state)
        else: 
            action = evader_policy(env, state)
        state, reward, done = env.step(action)
    return  reward, env.t

    
def evaluate(env, pursuer_policy, evader_policy, num_episodes=1000):
    caught_count = 0
    caught_steps = 0
    for n in range(num_episodes):
        reward, steps = run_episode(env, pursuer_policy, evader_policy)
        if reward == 1:
            caught_count += 1
            caught_steps += steps
    
        capture_rate = caught_count / num_episodes

        if caught_count > 0: 
            avg_steps = caught_steps / caught_count 
        else:
            float("nan")
    return capture_rate, avg_steps


if __name__ == "__main__":
    env_no = PursuitEvasionGrid(n=10, seed=0)
    env_hole = PursuitEvasionGrid(n=10, obstacles=[(4, 4), (4, 5)], seed=0)
    env_wall = PursuitEvasionGrid(n=10, obstacles=[(4, 0), (4, 1), (4, 2)], seed=0)
    env_set = [env_no, env_hole, env_wall]

    for env in env_set:
        rate, steps = evaluate(env, random_policy, random_policy)
        print(f"capture rate: {rate:.1%}")
        print(f"average steps to capture: {steps:.1f}")
    