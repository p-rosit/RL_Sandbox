import torch
from torch import nn
from torch import optim
import gymnasium as gym

import matplotlib.pyplot as plt

from buffer.experience_replay_buffer import ReplayBuffer
from buffer.online_buffer import OnlineEpisodeBuffer
from environment.q_learning_environment import QLearningEnvironment
from environment.policy_gradient_environment import PolicyGradientEnvironment

from networks.dense_networks import DenseEgoMotionQNetwork, DenseEgoMotionPolicyNetwork

from agents.random_agent import RandomAgent
from agents.off_policy.deep_q_learning.q_learning import QLearningAgent
from agents.off_policy.deep_q_learning.double_q_learning import DoubleQLearningAgent, ModifiedDoubleQLearningAgent, ClippedDoubleQLearning
from agents.off_policy.deep_q_learning.multi_q_learning import MultiQLearningAgent
from core.wrapper.agent_wrappers import AnnealAgent

from agents.on_policy.policy_gradient.policy_gradient import ReinforceAgent, ModifiedReinforceAgent
from agents.on_policy.policy_gradient.policy_gradient_baseline import ReinforceAdvantageAgent, ModifiedReinforceAdvantageAgent

def freeze(network):
    params = list(network.parameters())
    for param in params[:-1]:
        param.requires_grad = False

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    q_buffer = ReplayBuffer(max_size=50000)
    q_environment = QLearningEnvironment(env, q_buffer)
    p_buffer = OnlineEpisodeBuffer()
    p_environment = PolicyGradientEnvironment(env, p_buffer)

    pre_train = True
    pre_eps = 1000
    epochs = 3000
    b_size = 256
    q_nets = 0
    p_nets = 1

    input_size = 4
    hidden_sizes = [128, 128]
    output_size = 2

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

    # Gather experiences for pretraining on ego motion
    if pre_train:
        q_environment.train(r, pre_eps, batch_size, train_steps=1, eval_episodes=1)

    alpha_start = 10  # 100
    alpha_end = 0.1  # 1
    alpha_decay = 1000
    net_1 = DenseEgoMotionQNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)
    net_2 = DenseEgoMotionQNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)
    net_3 = DenseEgoMotionQNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)
    net_4 = DenseEgoMotionQNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)

    # q = QLearningAgent(net_1, discount=gamma, tau=tau)
    # q = DoubleQLearningAgent(net_1, net_2, discount=gamma, tau=tau, policy_train=False)
    # q = ModifiedDoubleQLearningAgent(net_1, discount=gamma, tau=tau)
    q = ClippedDoubleQLearning(net_1, net_2, discount=gamma, tau=tau)
    # q = MultiQLearningAgent(net_1, net_2, net_3, net_4, discount=gamma, tau=tau, policy_train=False)
    q.set_criterion(nn.SmoothL1Loss())
    q_optimizer = optim.AdamW(q.parameters(), lr=lr, amsgrad=True)
    q.set_optimizer(q_optimizer)

    sq = AnnealAgent(q, r, start_steps=start_steps, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay)

    num_rollouts = 5000

    net = DenseEgoMotionPolicyNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)

    # pn = ReinforceAgent(net, discount=gamma)
    # pn = ModifiedReinforceAgent(net, truncate_grad_trajectory=600, discount=gamma)
    pn = ReinforceAdvantageAgent(net, discount=gamma)
    # pn = ModifiedReinforceAdvantageAgent(net, truncate_grad_trajectory=600, discount=gamma)
    p_optimizer = optim.AdamW(pn.parameters(), lr=lr, amsgrad=True)
    pn.set_optimizer(p_optimizer)

    if pre_train:
        losss = []
        for _ in range(epochs):
            net_1.curr_step = 0
            net_2.curr_step = 0
            net_3.curr_step = 0
            net_4.curr_step = 0
            net.curr_step = 0

            experiences = q_buffer.sample(b_size)
            states, actions, rewards, next_states = list(zip(*experiences))
            states = [state for state, next_state in zip(states, next_states) if next_state is not None]
            actions = [action for action, next_state in zip(actions, next_states) if next_state is not None]
            next_states = [next_state for next_state in next_states if next_state is not None]

            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
            next_states = torch.cat(next_states, dim=0)

            loss = torch.zeros(1)
            if q_nets > 0:
                loss += net_1.intrinsic_loss(states, actions, None, next_states, torch.ones(next_states.size(0), dtype=torch.bool))
            if q_nets > 1:
                loss += net_2.intrinsic_loss(states, actions, None, next_states, torch.ones(next_states.size(0), dtype=torch.bool))
            if q_nets > 2:
                loss += net_3.intrinsic_loss(states, actions, None, next_states, torch.ones(next_states.size(0), dtype=torch.bool))
            if q_nets > 3:
                loss += net_4.intrinsic_loss(states, actions, None, next_states, torch.ones(next_states.size(0), dtype=torch.bool))

            if p_nets > 0:
                loss += net._intrinsic_loss(states, actions, next_states)

            losss.append(loss.item())

            q_optimizer.zero_grad()
            p_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()
            p_optimizer.step()

        plt.plot([l / b_size for l in losss])
        plt.show()

    freeze(net_1.initial_network)
    freeze(net_2.initial_network)
    freeze(net_3.initial_network)
    freeze(net_4.initial_network)
    freeze(net.initial_network)

    net_1.curr_step = 0
    net_2.curr_step = 0
    net_3.curr_step = 0
    net_4.curr_step = 0
    net.curr_step = 0

    # q_environment.train(sq, num_episodes, batch_size, train_steps=1, eval_episodes=1, plot=True)
    p_environment.train(pn, num_rollouts, train_steps=1, episodes_per_step=16, eval_episodes=10, plot=True)

    env.close()

if __name__ == '__main__':
    main()
