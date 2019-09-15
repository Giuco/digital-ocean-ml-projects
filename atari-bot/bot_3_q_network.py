"""
Bot 3 - Simple Q learning for Frozen Lake using nn
"""

import random
from typing import List

import gym
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

N_EPISODES = 5000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.15

REPORT_INTERVAL = 500


def exploration_probability(episode: int) -> float:
    return 50.0 / (episode + 10)


def one_hot_encode(i: int, n: int) -> np.ndarray:
    return np.identity(n)[i].reshape((1, -1))


def print_report(rewards: List[float], episode: int):
    last_100_avg = np.mean(rewards[-100:])
    avg = np.mean(rewards)
    max_100_average = np.max([np.mean(rewards[i:i + 100]) for i in range(len(rewards) - 100)])
    print(f"Episode: {episode} | 100-ep Average: {last_100_avg:.2f} | Average: {avg:.2f} | ",
          f"Best 100-ep Average: {max_100_average}")


def main():
    env = gym.make("FrozenLake-v0")
    env.seed(0)
    all_episodes_rewards = list()

    # 1. Setup placeholders
    n_obs, n_actions = env.observation_space.n, env.action_space.n
    obs_t_ph = tf.placeholder(shape=(1, n_obs), dtype=tf.float32)
    obs_tp1_ph = tf.placeholder(shape=(1, n_obs), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(), dtype=tf.int32)
    rew_ph = tf.placeholder(shape=(), dtype=tf.float32)
    q_target_ph = tf.placeholder(shape=(1, n_actions), dtype=tf.float32)

    # 2. Setup computation graph
    weights = tf.Variable(tf.random_uniform((n_obs, n_actions), 0, 0.01))
    q_current = tf.matmul(obs_t_ph, weights)
    q_target = tf.matmul(obs_tp1_ph, weights)

    q_target_max = tf.reduce_max(q_target_ph, axis=1)
    q_target_sa = rew_ph + DISCOUNT_FACTOR * q_target_max
    q_current_sa = q_current[0, act_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    pred_act_ph = tf.argmax(q_current, 1)

    # 3. Setup optimization
    trainer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    update_model = trainer.minimize(error)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for episode in range(1, N_EPISODES + 1):

            obs_t = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # 4. Take step using best action or random action
                obs_t_oh = one_hot_encode(obs_t, n_obs)

                if np.random.rand(1) < exploration_probability(episode):
                    action = env.action_space.sample()
                else:
                    action = session.run(pred_act_ph, feed_dict={obs_t_ph: obs_t_oh})[0]

                obs_tp1, reward, done, _ = env.step(action)

                # 5. Train model
                obs_tp1_oh = one_hot_encode(obs_tp1, n_obs)
                q_target_val = session.run(q_target, feed_dict={obs_tp1_ph: obs_tp1_oh})
                session.run(update_model, feed_dict={obs_t_ph: obs_t_oh,
                                                     rew_ph: reward,
                                                     q_target_ph: q_target_val,
                                                     act_ph: action})

                episode_reward += reward
                obs_t = obs_tp1

            all_episodes_rewards.append(episode_reward)
            if episode % REPORT_INTERVAL == 0:
                print_report(all_episodes_rewards, episode)


if __name__ == "__main__":
    main()
