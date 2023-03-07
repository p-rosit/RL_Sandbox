import torch
from torch import nn
from torch import optim
import gymnasium as gym

from buffer.online_buffer import OnlineEpisodeBuffer
from environment.policy_gradient_environment import PolicyGradientEnvironment

from networks.dense_networks import DensePolicyNetwork
from agents.on_policy.policy_gradient.policy_gradient import ReinforceAgent, ModifiedReinforceAgent
from agents.on_policy.policy_gradient.policy_gradient_baseline import ReinforceAdvantageAgent, ModifiedReinforceAdvantageAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = OnlineEpisodeBuffer()
    environment = PolicyGradientEnvironment(env, buffer)

    gamma = 0.99
    lr = 1e-4
    num_rollouts = 5000

    net = DensePolicyNetwork(4, [128, 128], 2)

    # pn = ReinforceAgent(net, discount=gamma)
    # pn = ModifiedReinforceAgent(net, truncate_grad_trajectory=600, discount=gamma)
    # pn = ReinforceAdvantageAgent(net, discount=gamma)
    pn = ModifiedReinforceAdvantageAgent(net, truncate_grad_trajectory=600, discount=gamma)
    pn.set_optimizer(optim.AdamW(pn.parameters(), lr=lr, amsgrad=True))

    environment.train(pn, num_rollouts, train_steps=1, episodes_per_step=16, eval_episodes=10, plot=True)

    env.close()

if __name__ == '__main__':
    main()
