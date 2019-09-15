"""
Bot 4 - Build least squares q-learning agent for FrozenLake
"""

import random
from typing import Callable, List, Tuple

import gym
import numpy as np

random.seed(0)
np.random.seed(0)

N_EPISODES = 6000
DISCOUNT_FACTOR = 0.85
LEARNING_RATE = 0.9
W_LR = 0.5

REPORT_INTERVAL = 500


def print_report(rewards: List[float], episode: int):
    last_100_avg = np.mean(rewards[-100:])
    avg = np.mean(rewards)
    max_100_average = np.max([np.mean(rewards[i:i + 100]) for i in range(len(rewards) - 100)])
    print(f"Episode: {episode} | 100-ep Average: {last_100_avg:.2f} | Average: {avg:.2f}",
          f"| Best 100-ep Average: {max_100_average}")


def make_q(model: np.array) -> Callable[[np.array], np.array]:
    return lambda x: x.dot(model)


def initialize_model(shape: Tuple) -> Tuple[np.array, Callable]:
    weights = np.random.normal(0.0, 0.1, shape)
    q = make_q(weights)
    return weights, q


def train_model(x: np.array, y: np.array, weights: np.array) -> Tuple[np.array, Callable]:
    i = np.eye(x.shape[1])
    new_weights = np.linalg.inv(x.T.dot(x) + 10e-4 * i).dot(x.T.dot(y))
    weights = W_LR * new_weights + (1 - W_LR) * weights
    q = make_q(weights)
    return weights, q


def one_hot(i: int, n: int) -> np.array:
    return np.identity(n)[i]


def main():
    env = gym.make("FrozenLake-v0")
    env.seed(0)
    all_episodes_rewards = list()

    n_obs, n_actions = env.observation_space.n, env.action_space.n
    weights, q = initialize_model((n_obs, n_actions))
    states, labels = [], []
    for episode in range(1, N_EPISODES + 1):
        if len(states) > 10_000:
            states, labels = [], []

        state_1 = one_hot(env.reset(), n_obs)
        done = False
        episode_reward = 0

        while not done:
            states.append(state_1)
            noise = np.random.random((1, n_actions)) / episode
            action = np.argmax(q(state_1) + noise)
            state_2, reward, done, _ = env.step(action)
            state_2 = one_hot(state_2, n_obs)

            q_target = reward + DISCOUNT_FACTOR * np.max(q(state_2))
            label = q(state_1)
            label[action] = (1 - LEARNING_RATE) * label[action] + LEARNING_RATE * q_target
            labels.append(label)
            episode_reward += reward
            state_1 = state_2

            if len(states) % 10 == 0:
                weights, q = train_model(np.array(states), np.array(labels), weights)

        all_episodes_rewards.append(episode_reward)
        if episode % REPORT_INTERVAL == 0:
            print_report(all_episodes_rewards, episode)


if __name__ == "__main__":
    main()
