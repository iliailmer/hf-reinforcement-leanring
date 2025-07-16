# TODO: Second simulatiuon

import gymnasium as gym
import torch.optim as optim
from loguru import logger

from .policy import Policy
from .reinforce import evaluate, reinforce
from .utils import push_to_hub

if __name__ == "__main__":
    env_id = "CartPole-v1"
    # Create the env
    env = gym.make(env_id, render_mode="rgb_array")

    # Create the evaluation env
    eval_env = gym.make(env_id, render_mode="rgb_array")

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = int(env.action_space.n)  # pyright: ignore

    logger.info("_____OBSERVATION SPACE_____ \n")
    logger.info("The State Space is: {}", s_size)
    logger.info(
        "Sample observation {}", env.observation_space.sample()
    )  # Get a random observation
    logger.info("\n _____ACTION SPACE_____ \n")
    logger.info("The Action Space is: {}", a_size)
    logger.info(
        "Action Space Sample {}", env.action_space.sample()
    )  # Take a random action
    cartpole_hyperparameters = {
        "h_size": 16,
        "n_training_episodes": 1000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }
    logger.info("Cartpole Hyperparameters:")
    logger.info(cartpole_hyperparameters)
    cartpole_policy = Policy(
        cartpole_hyperparameters["state_space"],
        cartpole_hyperparameters["action_space"],
        cartpole_hyperparameters["h_size"],
    ).to("mps")
    cartpole_optimizer = optim.Adam(
        cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"]
    )
    scores = reinforce(
        env,
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        print_every=100,
    )
    evaluate(
        eval_env,
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["n_evaluation_episodes"],
        cartpole_policy,
    )

    repo_id = "flyingeli4/Reinforce-cartpole"  # TODO Define your repo id {username/Reinforce-{model-id}}
    push_to_hub(
        repo_id,
        cartpole_policy,  # The model we want to save
        cartpole_hyperparameters,  # Hyperparameters
        eval_env,  # Evaluation environment
        video_fps=30,
    )
