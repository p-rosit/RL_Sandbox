import torch
from torch import nn
from torch import optim
import gymnasium as gym

from buffer.experience_replay_buffer import ReplayBuffer

from networks.DenseNetwork import DenseNetwork

from agents.random_agent import RandomAgent
from agents.off_policy.deep_q_learning.q_learning import QLearningAgent
from agents.off_policy.deep_q_learning.double_q_learning import DoubleQLearningAgent, ModifiedDoubleQLearningAgent, ClippedDoubleQLearning
from agents.off_policy.deep_q_learning.multi_q_learning import MultiQLearningAgent
from core.agent_wrappers import AnnealAgent

import matplotlib.pyplot as plt

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=50000)

    batch_size = 256
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    start_steps = 1000
    tau = 0.005
    lr = 1e-4
    num_episodes = 600

    r = RandomAgent(env)

    net_1 = DenseNetwork(4, [128, 128], 2)
    net_2 = DenseNetwork(4, [128, 128], 2)
    net_3 = DenseNetwork(4, [128, 128], 2)
    net_4 = DenseNetwork(4, [128, 128], 2)

    # q = QLearningAgent(net_1, discount=gamma, tau=tau)
    # q = DoubleQLearningAgent(net_1, net_2, discount=gamma, tau=tau, policy_train=False)
    # q = ModifiedDoubleQLearningAgent(net_1, discount=gamma, tau=tau)
    q = ClippedDoubleQLearning(net_1, net_2, discount=gamma, tau=tau)
    # q = MultiQLearningAgent(net_1, net_2, net_3, net_4, discount=gamma, tau=tau, policy_train=False)
    q.set_criterion(nn.SmoothL1Loss())
    q.set_optimizer(optim.AdamW(q.parameters(), lr=lr, amsgrad=True))

    sq = AnnealAgent(q, r, start_steps=start_steps, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay)

    episode_reward = []
    evaluation_episode = []
    evaluation_reward = []
    fig = plt.figure(1)
    ax = fig.subplots()

    for ep in range(num_episodes):
        done = False
        curr_reward = 0
        state, info = env.reset()

        if ep % 10 == 0:
            # Run evaluation episode
            sq.eval()
            while not done:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                _, env_action = sq.sample(state)
                observation, reward, terminated, truncated, _ = env.step(env_action)
                curr_reward += reward
                done = terminated or truncated

                if done:
                    break

                state = observation

            evaluation_episode.append(ep)
            evaluation_reward.append(curr_reward)

            sq.train()
            done = False
            curr_reward = 0
            state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while not done:
            policy_action, env_action = sq.sample(state)
            observation, reward, terminated, truncated, _ = env.step(env_action)
            curr_reward += reward
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            buffer.append(state, policy_action, reward, next_state)
            state = next_state

            if len(buffer) > batch_size:
                sq.step(buffer.sample(batch_size))

            if done:
                break

        episode_reward.append(curr_reward)

        ax.cla()
        ax.plot(episode_reward)
        ax.plot(evaluation_episode, evaluation_reward)
        plt.draw()
        plt.pause(0.0001)

    env.close()

if __name__ == '__main__':
    main()
