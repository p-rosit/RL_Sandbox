import torch
from torch import nn
from torch import optim
import gymnasium as gym

from experience_replay_buffer import ReplayBuffer

from random_actor import RandomActor
from q_learning import DenseQLearningActor
from actor_wrappers import Discrete2Continuous, MultiActor, AnnealActor

import matplotlib.pyplot as plt
import numpy as np

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=10000)

    actor = RandomActor(env)
    q = DenseQLearningActor(4, [128], 2, discount=0.99)
    # qa = Discrete2Continuous(q, remap=[0, 1, 2, 3])
    sq = AnnealActor(q, actor)
    # sq = MultiActor(actor, q, p=[0.01, 0.99])

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    # optimizer = optim.SGD(qa.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(q.parameters(), lr=1e-4, amsgrad=True)
    q.set_criterion(criterion)
    q.set_optimizer(optimizer)

    episode_reward = []
    episode_length = []
    r = 0
    fig = plt.figure(1)
    ax = fig.subplots()

    step = 0
    prev_observation, info = env.reset(seed=42)
    for _ in range(100000):
        step += 1
        # action = env.action_space.sample()  # this is where you would insert your policy
        # print(action)
        # action = actor.sample(prev_observation)
        # action = sa.sample(prev_observation)
        action = sq.sample(prev_observation)
        observation, reward, terminated, truncated, info = env.step(action)

        buffer.append(prev_observation, action, reward, observation)

        r += reward

        if len(buffer) > 128:
            # for _ in range(1):
            sq.step(buffer.sample(10))

        if terminated or truncated:
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
