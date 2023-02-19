import gymnasium as gym
from experience_replay_buffer import ReplayBuffer
from random_actor import RandomActor
from q_learning import DenseQLearningActor
from actor_wrappers import Discrete2Continuous, MultiActor


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    buffer = ReplayBuffer(max_size=100)

    actor = RandomActor(env)
    q = DenseQLearningActor(8, [10], 3, discount=0.99)
    qa = Discrete2Continuous(q, remap=[0, 1, 2, 3])
    sq = MultiActor(actor, qa, p=[0.5, 0.5])

    prev_observation, info = env.reset(seed=42)
    for _ in range(100):
        # action = env.action_space.sample()  # this is where you would insert your policy
        # action = actor.sample(prev_observation)
        # action = sa.sample(prev_observation)
        action = sq.sample(prev_observation)
        observation, reward, terminated, truncated, info = env.step(action)

        buffer.append(prev_observation, action, reward, observation)

        if len(buffer) > 10:
            sq.step(buffer.sample(5))

        if terminated or truncated:
            observation, info = env.reset()
        prev_observation = observation
    env.close()


if __name__ == '__main__':
    main()
