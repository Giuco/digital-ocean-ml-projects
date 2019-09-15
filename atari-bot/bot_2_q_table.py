"""
Bot 2 - Simple Q learning for Frozen Lake
"""

import random
from typing import List

import gym
import numpy as np

random.seed(0)
np.random.seed(0)

N_EPISODES = 5000
DISCOUNT_FACTOR = 0.8
LEARNING_RATE = 0.9

REPORT_INTERVAL = 500


def print_report(rewards: List[float], episode: int):
    last_100_avg = np.mean(rewards[-100:])
    avg = np.mean(rewards)
    max_100_average = np.max([np.mean(rewards[i:i + 100]) for i in range(len(rewards) - 100)])
    print(f"Episode: {episode} | 100-ep Average: {last_100_avg:.2f} | Average: {avg:.2f}",
          f"| Best 100-ep Average: {max_100_average}")


def run_episode(episode, env, q_table):
    state_1 = env.reset()
    done = False
    episode_reward = 0

    while not done:
        noise = np.random.random((1, env.action_space.n)) / (episode ** 2.0)
        action = np.argmax(q_table[state_1, :] + noise)
        state_2, reward, done, _ = env.step(action)
        q_target = reward + DISCOUNT_FACTOR * np.max(q_table[state_2, :])
        q_table[state_1, action] = (1 - LEARNING_RATE) * q_table[state_1, action] + LEARNING_RATE * q_target
        episode_reward += reward
        state_1 = state_2

    return q_table, episode_reward


def main():
    env = gym.make("FrozenLake-v0")
    env.seed(0)
    all_episodes_rewards = list()

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(1, N_EPISODES + 1):
        q_table, episode_reward = run_episode(episode, env, q_table)
        all_episodes_rewards.append(episode_reward)
        if episode % REPORT_INTERVAL == 0:
            print_report(all_episodes_rewards, episode)


if __name__ == "__main__":
    main()
