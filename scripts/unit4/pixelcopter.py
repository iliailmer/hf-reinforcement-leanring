# NOTE: could not get this to work on macos
import gymnasium as gym
import gym_minatar
import gym_pygame
import gym_exploration

import torch.optim as optim
from loguru import logger

from .policy import PolicyV2
from .reinforce import evaluate, reinforce
from .utils import push_to_hub

if __name__ == "__main__":
    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = int(env.action_space.n)  # pyright: ignore
    logger.info("_____OBSERVATION SPACE_____ \n")
    logger.info("The State Space is: {}", s_size)
    logger.info("Sample observation {}", env.observation_space.sample())
    logger.info("\n _____ACTION SPACE_____ \n")
    logger.info("The Action Space is: {}", a_size)
    logger.info("Action Space Sample {}", env.action_space.sample())
    copter_hyperparameters = {
        "h_size": 16,
        "n_training_episodes": 10,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }
    logger.info("Copter Hyperparameters:")
    logger.info(copter_hyperparameters)
    copter_policy = PolicyV2(
        copter_hyperparameters["state_space"],
        copter_hyperparameters["action_space"],
        copter_hyperparameters["h_size"],
    ).to("mps")
    copter_optimizer = optim.Adam(
        copter_policy.parameters(), lr=copter_hyperparameters["lr"]
    )
    scores = reinforce(
        env,
        copter_policy,
        copter_optimizer,
        copter_hyperparameters["n_training_episodes"],
        copter_hyperparameters["max_t"],
        copter_hyperparameters["gamma"],
        print_every=100,
    )
    evaluate(
        eval_env,
        copter_hyperparameters["max_t"],
        copter_hyperparameters["n_evaluation_episodes"],
        copter_policy,
    )

    repo_id = "flyingeli4/Reinforce-pixelcopter"  # TODO Define your repo id {username/Reinforce-{model-id}}
    push_to_hub(
        repo_id,
        copter_policy,  # The model we want to save
        copter_hyperparameters,  # Hyperparameters
        eval_env,  # Evaluation environment
        video_fps=30,
    )
