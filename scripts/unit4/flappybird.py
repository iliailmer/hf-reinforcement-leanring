import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.optim as optim
from loguru import logger

from .policy import PolicyCNN, PolicyV2
from .reinforce import evaluate, evaluate_conv, reinforce, reinforce_conv
from .utils import push_to_hub, record_video

if __name__ == "__main__":
    env_id = "FlappyBird-v0"
    # Create the env
    env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)

    # Create the evaluation env
    eval_env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = int(env.action_space.n)  # pyright: ignore

    logger.info("_____OBSERVATION SPACE_____ \n")
    logger.info("The State Space is: {}", s_size)
    logger.info("Sample observation {}", env.observation_space.sample())
    logger.info("\n _____ACTION SPACE_____ \n")
    logger.info("The Action Space is: {}", a_size)
    logger.info("Action Space Sample {}", env.action_space.sample())
    bird_hyperparameters = {
        "h_size": 16,
        "n_training_episodes": 5000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "n_frames": 8,
        "gamma": 0.95,
        "lr": 5e-5,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    policy = PolicyCNN(
        in_channels=3 * bird_hyperparameters["n_frames"],
        h_size=bird_hyperparameters["h_size"],
        a_size=a_size,
        state_dim=s_size,
    ).to("mps")

    optimizer = optim.AdamW(policy.parameters(), lr=bird_hyperparameters["lr"])

    scores = reinforce_conv(
        env,
        policy,
        optimizer=optimizer,
        n_training_episodes=bird_hyperparameters["n_training_episodes"],
        max_t=bird_hyperparameters["max_t"],
        gamma=bird_hyperparameters["gamma"],
        n_frames=bird_hyperparameters["n_frames"],
        print_every=100,
    )
    eval_results = evaluate_conv(
        eval_env,
        bird_hyperparameters["max_t"],
        bird_hyperparameters["n_evaluation_episodes"],
        policy=policy,
        n_frames=bird_hyperparameters["n_frames"],
    )
    repo_id = "flyingeli4/Reinforce-flappybird"  # TODO Define your repo id {username/Reinforce-{model-id}}
    record_video(eval_env, policy, "./")
    push_to_hub(
        repo_id,
        policy,  # The model we want to save
        bird_hyperparameters,  # Hyperparameters
        eval_env,  # Evaluation environment
        video_fps=30,
    )
