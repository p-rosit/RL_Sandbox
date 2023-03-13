from torch import optim
import gymnasium as gym

from buffer.experience_replay_buffer import ReplayBuffer
from environment.actor_critic_environment import ActorCriticEnvironment

from networks.dense_networks import DenseActorCriticNetwork

from agents.on_policy.actor_critic.actor_critic import ActorCriticAgent

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(max_size=50000)
    environment = ActorCriticEnvironment(env, buffer)

    input_size = 4
    actor_hidden_sizes = [128, 128]
    critic_hidden_sizes = [128, 128]
    output_size = 2

    gamma = 0.99
    lr = 1e-4
    num_episodes = 600

    initial_episodes = 1000
    epochs = 1000
    pre_batch = 1000

    net = DenseActorCriticNetwork(input_size, actor_hidden_sizes, output_size, critic_hidden_sizes)

    ac = ActorCriticAgent(net, discount=gamma)
    optimizer = optim.AdamW(ac.parameters(), lr=lr, amsgrad=True)

    # environment.explore(RandomAgent(env), initial_episodes)
    # environment.pretrain(ac, optimizer, epochs, pre_batch, plot=True)
    # environment.buffer.clear()
    environment.train(ac, optimizer, num_episodes, train_steps=1, eval_episodes=1, td_steps=3, plot=True)

    env.close()

if __name__ == '__main__':
    main()