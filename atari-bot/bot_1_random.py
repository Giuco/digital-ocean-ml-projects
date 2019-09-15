"""
Bot 1 - Random baseline for Space Invaders
"""

import gym
import random
from statistics import mean

random.seed(0)

N_EPISODES = 10


def run_random_episode(env) -> float:
    env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        episode_reward += reward

    print(f'Reward: {episode_reward}')
    return episode_reward


def main():
    env = gym.make("SpaceInvaders-v0")
    env.seed(0)
    all_episodes_rewards = list()

    for _ in range(N_EPISODES):
        episode_reward = run_random_episode(env)
        all_episodes_rewards.append(episode_reward)

    print(f"Mean reward {mean(all_episodes_rewards)}")


if __name__ == "__main__":
    main()
