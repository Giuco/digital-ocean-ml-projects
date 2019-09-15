"""
Bot 5 - Fully featured deep q-learning network.
"""
import random

import cv2
import gym
import numpy as np
import tensorflow as tf

from bot_5_a3c import a3c_model

random.seed(0)
tf.set_random_seed(0)

N_EPISODES = 10


def downsample(state):
    return cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)[None]


def main():
    env = gym.make('SpaceInvaders-v0')
    env.seed(0)
    rewards = []
    model = a3c_model(load='models/SpaceInvaders-v0.tfmodel')

    for _ in range(N_EPISODES):
        episode_reward = 0
        states = [downsample(env.reset())]
        done = False

        while not done:
            if len(states) < 4:
                action = env.action_space.sample()
            else:
                frames = np.concatenate(states[-4:], axis=3)
                action = np.argmax(model([frames]))

            state, reward, done, _ = env.step(action)
            states.append(downsample(state))
            episode_reward += reward

        print(f"Episode reward: {episode_reward}")
        rewards.append(episode_reward)

    print(f"Average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
