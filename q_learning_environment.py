import torch
from torch import nn
from torch import optim
import gymnasium as gym

from experience_replay_buffer import ReplayBuffer

from random_actor import RandomActor
from q_learning import DenseQLearningActor
from actor_wrappers import AnnealActor

import matplotlib.pyplot as plt
import numpy as np

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=10000)

    batch_size = 128
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    tau = 0.005
    lr = 1e-4

    actor = RandomActor(env)

    q = DenseQLearningActor(4, [128], 2, discount=gamma, tau=tau)
    q.set_criterion(nn.SmoothL1Loss())
    q.set_optimizer(optim.AdamW(q.parameters(), lr=lr, amsgrad=True))

    sq = AnnealActor(q, actor, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay)

    episode_reward = []
    episode_length = []
    r = 0
    fig = plt.figure(1)
    ax = fig.subplots()

    step = 0
    prev_observation, info = env.reset(seed=42)
    for _ in range(100000):
        step += 1
        action = sq.sample(prev_observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if terminated:
            observation = None

        buffer.append(prev_observation, action, reward, observation)

        r += reward

        if len(buffer) > batch_size:
            sq.step(buffer.sample(batch_size))

        if done:
            episode_length.append(step)
            step = 0

            episode_reward.append(r)
            ax.cla()
            # ax.plot(np.convolve(episode_reward, np.ones(10) / 10, 'valid'))
            ax.plot(np.convolve(episode_length, np.ones(10) / 10, 'valid'))
            plt.draw()
            plt.pause(0.001)

            observation, info = env.reset()
            r = 0

        prev_observation = observation
    env.close()

if __name__ == '__main__':
    main()
