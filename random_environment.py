import gymnasium as gym

from agents.random_agent import RandomAgent

import matplotlib.pyplot as plt

def main():
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")

    num_episodes = 600

    r = RandomAgent(env)

    episode_reward = []
    fig = plt.figure(1)
    ax = fig.subplots()

    for _ in range(num_episodes):
        done = False
        step = 0
        curr_reward = 0
        env.reset()

        while not done:
            step += 1
            action = r.sample(None)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            curr_reward += reward
            done = terminated or truncated

            if done:
                episode_reward.append(curr_reward)

                ax.cla()
                ax.plot(episode_reward)
                plt.draw()
                plt.pause(0.0001)
                break

    env.close()

if __name__ == '__main__':
    main()
