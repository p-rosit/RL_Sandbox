from torch import nn
from torch import optim
import gymnasium as gym

from buffer.online_buffer import OnlineEpisodeBuffer
from environment.policy_gradient_environment import PolicyGradientEnvironment

from networks.dense_networks import DensePolicyNetwork
from agents.on_policy.policy_gradient.reinforce import ReinforceAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = OnlineEpisodeBuffer()
    environment = PolicyGradientEnvironment(env, buffer)

    gamma = 0.99
    lr = 1e-4
    num_episodes = 600

    net = DensePolicyNetwork(4, [128, 128], 2)
    pn = ReinforceAgent(net, discount=gamma)

    environment.train(pn, num_episodes, episodes_per_step=1, train_steps=1, eval_episodes=1, plot=True)

    env.close()

if __name__ == '__main__':
    main()
