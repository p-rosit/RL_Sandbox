import torch
from torch import nn
from torch import optim
import gymnasium as gym

from experience_replay_buffer import ReplayBuffer

from random_actor import RandomActor
from q_learning import DenseQLearningActor
from actor_wrappers import AnnealActor

import matplotlib.pyplot as plt

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=10000)

    batch_size = 128
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    start_steps = 1000
    tau = 0.005
    lr = 1e-4
    num_episodes = 10000

    r = RandomActor(env)

    q = DenseQLearningActor(4, [128, 128], 2, discount=gamma, tau=tau)
    q.set_criterion(nn.SmoothL1Loss())
    q.set_optimizer(optim.AdamW(q.parameters(), lr=lr, amsgrad=True))

    sq = AnnealActor(q, r, start_steps=start_steps, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay)

    episode_length = []
    fig = plt.figure(1)
    ax = fig.subplots()

    for _ in range(num_episodes):
        done = False
        step = 0
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while not done:
            step += 1
            action = sq.sample(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            buffer.append(state, action, reward, next_state)
            state = next_state

            if len(buffer) > batch_size:
                sq.step(buffer.sample(batch_size))

            if done:
                episode_length.append(step + 1)

                ax.cla()
                ax.plot(episode_length)
                plt.draw()
                plt.pause(0.0001)
                break

    env.close()

if __name__ == '__main__':
    main()
