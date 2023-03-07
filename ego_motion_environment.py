from torch import nn
from torch import optim
import gymnasium as gym

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

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=50000)
    q_environment = QLearningEnvironment(env, buffer)
    buffer = OnlineEpisodeBuffer()
    p_environment = PolicyGradientEnvironment(env, buffer)

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
    q.set_optimizer(optim.AdamW(q.parameters(), lr=lr, amsgrad=True))

    sq = AnnealAgent(q, r, start_steps=start_steps, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay)

    num_rollouts = 5000

    net = DenseEgoMotionPolicyNetwork(input_size, hidden_sizes, output_size, alpha_start=alpha_start, alpha_end=alpha_end, alpha_decay=alpha_decay)

    # pn = ReinforceAgent(net, discount=gamma)
    # pn = ModifiedReinforceAgent(net, truncate_grad_trajectory=600, discount=gamma)
    pn = ReinforceAdvantageAgent(net, discount=gamma)
    # pn = ModifiedReinforceAdvantageAgent(net, truncate_grad_trajectory=600, discount=gamma)
    pn.set_optimizer(optim.AdamW(pn.parameters(), lr=lr, amsgrad=True))

    # q_environment.train(sq, num_episodes, batch_size, train_steps=1, eval_episodes=1, plot=True)
    p_environment.train(pn, num_rollouts, train_steps=1, episodes_per_step=16, eval_episodes=10, plot=True)

    env.close()

if __name__ == '__main__':
    main()
