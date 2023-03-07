import matplotlib.pyplot as plt
import torch

class PolicyGradientEnvironment:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer

    def train(self, agent, num_rollouts, episodes_per_step=1, eval_episodes=0, plot=False):
        episode_reward = []
        evaluation_episode = []
        evaluation_reward = []
        fig = plt.figure(1) if plot else None
        ax = fig.subplots() if fig is not None else None

        for roll in range(num_rollouts):
            if roll % 10 == 0 and eval_episodes > 0:
                agent.eval()
                evaluation_episode.append(roll)
                evaluation_reward.append(self.eval(agent, eval_episodes))
                agent.train()

            curr_reward = 0
            for _ in range(episodes_per_step):
                done = False
                state, info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                while not done:
                    policy_action, env_action = agent.sample(state)
                    observation, reward, terminated, truncated, _ = self.env.step(env_action)
                    curr_reward += reward
                    reward = torch.tensor([reward])
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                    self.buffer.append(state, policy_action, reward, episode_terminated=done)
                    state = next_state

            agent.step(self.buffer.sample())
            self.buffer.clear()

            episode_reward.append(curr_reward / episodes_per_step)

            if plot:
                ax.cla()
                ax.plot(episode_reward)
                ax.plot(evaluation_episode, evaluation_reward)
                plt.draw()
                plt.pause(0.0001)

    def eval(self, agent, num_episodes):
        mean_reward = 0
        for _ in range(num_episodes):
            done = False
            observation, info = self.env.reset()

            while not done:
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                _, env_action = agent.sample(state)
                observation, reward, terminated, truncated, _ = self.env.step(env_action)
                mean_reward += reward
                done = terminated or truncated

        return mean_reward / num_episodes
