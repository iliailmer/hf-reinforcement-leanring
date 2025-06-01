import argparse
import json
import random

import gymnasium as gym
import numpy as np
from utils import evaluate_agent, initialize_q_table, push_to_hub, train

np.random.seed(42)
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-Learning agent parameters for FrozenLake-v1"
    )

    parser.add_argument(
        "--n_training_episodes",
        type=int,
        default=10000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.7, help="Learning rate (alpha)"
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env_id", type=str, default="FrozenLake-v1", help="Gym environment ID"
    )
    parser.add_argument(
        "--map_name",
        type=str,
        default="4x4",
        help="Map name for FrozenLake (4x4 or 8x8)",
    )
    parser.add_argument(
        "--slippery", action="store_true", help="Whether the environment is slippery"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        help="Render mode (default: rgb_array)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=99, help="Maximum number of steps per episode"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--max_epsilon", type=float, default=1.0, help="Initial exploration rate"
    )
    parser.add_argument(
        "--min_epsilon", type=float, default=0.05, help="Final exploration rate"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.0005, help="Exploration decay rate"
    )
    parser.add_argument(
        "--username", type=str, default="flyingeli4", help="Your HuggingFace username"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="q-FrozenLake-v1-4x4-noSlippery",
        help="HuggingFace repo name",
    )
    parser.add_argument(
        "--save_config_path",
        type=str,
        default="config.json",
        help="Path to save the config as JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_seed = []
    if args.env_id == "Taxi-v3":
        eval_seed = [
            16,
            54,
            165,
            177,
            191,
            191,
            120,
            80,
            149,
            178,
            48,
            38,
            6,
            125,
            174,
            73,
            50,
            172,
            100,
            148,
            146,
            6,
            25,
            40,
            68,
            148,
            49,
            167,
            9,
            97,
            164,
            176,
            61,
            7,
            54,
            55,
            161,
            131,
            184,
            51,
            170,
            12,
            120,
            113,
            95,
            126,
            51,
            98,
            36,
            135,
            54,
            82,
            45,
            95,
            89,
            59,
            95,
            124,
            9,
            113,
            58,
            85,
            51,
            134,
            121,
            169,
            105,
            21,
            30,
            11,
            50,
            65,
            12,
            43,
            82,
            145,
            152,
            97,
            106,
            55,
            31,
            85,
            38,
            112,
            102,
            168,
            123,
            97,
            21,
            83,
            158,
            26,
            80,
            63,
            5,
            81,
            32,
            11,
            28,
            148,
        ]
    # Save config to JSON file
    with open(args.save_config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.env_id == "FrozenLake-v1":
        env = gym.make(
            args.env_id,
            map_name=args.map_name,
            is_slippery=args.slippery,
            render_mode=args.render_mode,
        )
    else:
        env = gym.make(args.env_id, render_mode=args.render_mode)

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
        args.n_training_episodes,
        args.min_epsilon,
        args.max_epsilon,
        args.decay_rate,
        env,
        args.max_steps,
        q_table,
        args.learning_rate,
        args.gamma,
    )
    print(q_table)

    mean_reward, std_reward = evaluate_agent(
        env, args.max_steps, args.n_eval_episodes, q_table, eval_seed
    )
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    model = {
        "env_id": args.env_id,
        "map_name": args.map_name,
        "is_slippery": args.slippery,
        "render_mode": args.render_mode,
        "max_steps": args.max_steps,
        "n_training_episodes": args.n_training_episodes,
        "n_eval_episodes": args.n_eval_episodes,
        "eval_seed": eval_seed,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "max_epsilon": args.max_epsilon,
        "min_epsilon": args.min_epsilon,
        "decay_rate": args.decay_rate,
        "qtable": q_table,
    }

    push_to_hub(repo_id=f"{args.username}/{args.repo_name}", model=model, env=env)
