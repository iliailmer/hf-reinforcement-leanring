import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
from gymnasium.wrappers import RecordVideo

import torch
import random
import numpy as np

SEED = 42
model_name = "ppo-LunarLander-v2"
env_id = "LunarLander-v3"
video_folder = "./videos/"
repo_id = "flyingeli4/lunar-lander-lesson-1"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# parameters

learning_rate = 1.5e-4  # get_linear_fn(3e-4, 1e-6, 0.05)
n_envs = 32
n_steps = int(2e6)
batch_size = 64
n_epochs = 20

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    env.reset(seed=SEED)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    env = make_vec_env(env_id, n_envs=n_envs, seed=SEED)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=SEED,
    )
    model.learn(total_timesteps=n_steps, progress_bar=True)
    model.save(model_name)

    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env, video_folder=video_folder, episode_trigger=lambda _: True
    )

    obs, _ = eval_env.reset(seed=SEED)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    eval_env.close()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )

    print(mean_reward, std_reward)

    # Create the evaluation env and set the render_mode="rgb_array"
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    model_architecture = "PPO"
    commit_message = "trying to upload v3 model from script"

    # method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
    package_to_hub(
        model=model,  # Our trained model
        model_name=model_name,  # The name of our trained model
        model_architecture=model_architecture,  # The model architecture we used: in our case PPO
        env_id=env_id,  # Name of the environment
        eval_env=eval_env,  # Evaluation Environment
        repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
        commit_message=commit_message,
    )
