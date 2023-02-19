import gymnasium as gym
from experience_replay_buffer import ReplayBuffer
from random_actor import RandomActor


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    buffer = ReplayBuffer(max_size=100)

    actor = RandomActor(env)

    prev_observation, info = env.reset(seed=42)
    for _ in range(1000):
        # action = env.action_space.sample()  # this is where you would insert your policy
        action = actor.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        buffer.append(prev_observation, action, reward, observation)

        if terminated or truncated:
            observation, info = env.reset()
        prev_observation = observation
    env.close()


if __name__ == '__main__':
    main()
