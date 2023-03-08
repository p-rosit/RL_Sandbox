from torch import nn
from torch import optim
import gymnasium as gym

from buffer.experience_replay_buffer import ReplayBuffer
from environment.q_learning_environment import QLearningEnvironment

from networks.dense_networks import DenseQNetwork, DenseEgoMotionQNetwork

from pre_training.ego_motion import pre_train_state_dynamics

from agents.random_agent import RandomAgent
from agents.off_policy.deep_q_learning.q_learning import QLearningAgent
from agents.off_policy.deep_q_learning.double_q_learning import DoubleQLearningAgent, ModifiedDoubleQLearningAgent, ClippedDoubleQLearning
from agents.off_policy.deep_q_learning.multi_q_learning import MultiQLearningAgent
from core.wrapper.agent_wrappers import AnnealAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=50000)
    environment = QLearningEnvironment(env, buffer)

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

    initial_episodes = 1000
    epochs = 1000
    ego_batch = 1024

    # net_1 = DenseQNetwork(input_size, hidden_sizes, output_size)
    # net_2 = DenseQNetwork(input_size, hidden_sizes, output_size)
    # net_3 = DenseQNetwork(input_size, hidden_sizes, output_size)
    # net_4 = DenseQNetwork(input_size, hidden_sizes, output_size)

    alpha_start = 10
    alpha_end = 0.00000001
    alpha_decay = 10
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

    environment.train(r, initial_episodes, 1)
    for net in (net_1, net_2, net_3, net_4):
        pre_train_state_dynamics(
            net,
            optim.AdamW(net.parameters(), lr=lr, amsgrad=True),
            buffer,
            epochs,
            ego_batch,
            plot=False
        )
    buffer.clear()

    environment.train(sq, num_episodes, batch_size, train_steps=1, eval_episodes=1, plot=True)

    env.close()

if __name__ == '__main__':
    main()
