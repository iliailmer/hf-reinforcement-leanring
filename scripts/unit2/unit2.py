import random

from utils import initialize_q_table, train, evaluate_agent, push_to_hub
import gymnasium as gym
import numpy as np

np.random.seed(42)
random.seed(42)


n_training_episodes = 10000
learning_rate = 0.7


n_eval_episodes = 100


env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []


max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
    )
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample())
    print("\n_____ACTION SPACE_____\n")
    print("Action Space Shape", env.action_space.n)  # pyright: ignore
    print("Action Space Sample", env.action_space.sample())

    print("\n** STEP 1: INITIALIZE Q-TABLE**")
    state_space = env.observation_space.n  # pyright: ignore
    print("There are", state_space, "possible states")
    action_space = env.action_space.n  # pyright: ignore
    print("There are", action_space, "possible actions")
    q_table = initialize_q_table(state_space, action_space)

    print("\n** STEPS 2-4: TRAIN THE MODEL (Q-LEARNING)")
    q_table = train(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        q_table,
        learning_rate,
        gamma,
    )
    print(q_table)

    mean_reward, std_reward = evaluate_agent(
        env, max_steps, n_eval_episodes, q_table, eval_seed
    )
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    model = {
        "env_id": env_id,
        "max_steps": max_steps,
        "n_training_episodes": n_training_episodes,
        "n_eval_episodes": n_eval_episodes,
        "eval_seed": eval_seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "max_epsilon": max_epsilon,
        "min_epsilon": min_epsilon,
        "decay_rate": decay_rate,
        "qtable": q_table,
    }
    username = "flyingeli4"
    repo_name = "q-FrozenLake-v1-4x4-noSlippery"
    push_to_hub(repo_id=f"{username}/{repo_name}", model=model, env=env)
