import torch
from torch import optim
import gymnasium as gym

from buffer.experience_replay_buffer import PrioritizedExperienceReplayBuffer
from environment.policy_gradient_environment import PolicyGradientEnvironment

from networks.policy_gradient.dense_networks import DensePolicyNetwork, DenseEgoMotionPolicyNetwork
from agents.random_agent import RandomAgent
from agents.on_policy.policy_gradient.policy_gradient import ReinforceAgent, ModifiedReinforceAgent
from agents.on_policy.policy_gradient.policy_gradient_baseline import ReinforceAdvantageAgent, ModifiedReinforceAdvantageAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = PrioritizedExperienceReplayBuffer(max_size=torch.inf)
    environment = PolicyGradientEnvironment(env, buffer)

    input_size = 4
    hidden_sizes = [128, 128]
    output_size = 2

    gamma = 0.99
    lr = 1e-4
    num_rollouts = 5000

    initial_episodes = 1000
    epochs = 1000
    pre_batch = 1000

    # net = DensePolicyNetwork(input_size, hidden_sizes, output_size)

    alpha_start = 0
    net = DenseEgoMotionPolicyNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start)

    # pn = ReinforceAgent(net, discount=gamma)
    # pn = ModifiedReinforceAgent(net, truncate_grad_trajectory=600, discount=gamma)
    pn = ReinforceAdvantageAgent(net, discount=gamma)
    # pn = ModifiedReinforceAdvantageAgent(net, truncate_grad_trajectory=600, discount=gamma)
    optimizer = optim.AdamW(pn.parameters(), lr=lr, amsgrad=True)

    # environment.explore(RandomAgent(env), initial_episodes)
    # environment.pretrain(pn, optimizer, epochs, pre_batch, plot=True)
    # environment.buffer.clear()
    environment.train(pn, optimizer, num_rollouts, train_steps=1, episodes_per_rollout=256, eval_episodes=10, plot=True)

    env.close()

if __name__ == '__main__':
    main()
