from collections import deque

import numpy as np
import torch
from loguru import logger
from torch import optim

from .policy import Policy


@logger.catch
def reinforce(
    env,
    policy: Policy,
    optimizer: optim.Optimizer,
    n_training_episodes: int,
    max_t: int,
    gamma: float,
    print_every: int,
):
    scores_dq = deque(maxlen=100)
    scores = []
    for i in range(n_training_episodes):
        # Generate an episode
        state, _ = env.reset()
        rewards = []

        episode_log_probas = []
        for t in range(max_t):
            action, log_proba = policy.act(state=state)
            episode_log_probas.append(log_proba)
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_dq.append(sum(rewards))
        scores.append(sum(rewards))
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(return_t * gamma + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (eps + returns.std())
        loss = torch.tensor(0.0)
        for t in range(n_steps):
            loss += -episode_log_probas[t].squeeze() * returns[t]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_every == 0:
            logger.info(
                f"Episode: {i}\t Average Score: {np.mean(scores_dq):.2f}\tLoss: {loss.item():.2f}"
            )

    return scores


@logger.catch
def evaluate(env, max_eval_steps, n_eval_episodes, policy: Policy):
    rewards = []
    for ep in range(n_eval_episodes):
        state, _ = env.reset()[0]
        episode_rewards = 0
        for step in range(max_eval_steps):
            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            if terminated or truncated:
                break
            state = new_state
            rewards.append(episode_rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"EVAL: Mean reward {mean_reward:.2f}+/-{std_reward:.2f}")
    return mean_reward, std_reward
