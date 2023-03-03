from torch import nn
from torch import optim
import gymnasium as gym

from buffer.online_buffer import OnlineEpisodeBuffer
from environment.policy_gradient_environment import PolicyGradientEnvironment
from agents.random_agent import RandomAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = OnlineEpisodeBuffer()
    environment = PolicyGradientEnvironment(env, buffer)

    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    start_steps = 1000
    lr = 1e-4
    num_episodes = 600

    r = RandomAgent(env)

    environment.train(r, num_episodes, episodes_per_step=1, train_steps=1, eval_episodes=1, plot=True)

    env.close()

if __name__ == '__main__':
    main()
