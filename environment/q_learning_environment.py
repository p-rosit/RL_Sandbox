import matplotlib.pyplot as plt
import torch

class QLearningEnvironment:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer

    def explore(self, agent, num_episodes):
        for _ in range(num_episodes):
            done = False
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            while not done:
                policy_action, env_action = agent.sample(state)
                observation, reward, terminated, truncated, _ = self.env.step(env_action)
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # self.buffer.append(state, policy_action, reward, next_state)
                self.buffer.append(state, policy_action, reward, episode_terminated=done)
                state = next_state

    def pretrain(self, agent, epochs, batch_size, plot=False):
        history = []

        for _ in range(epochs):
            loss = agent.pretrain_loss(self.buffer.sample(batch_size))

            history.append(loss.item())

            agent.optimizer_zero_grad()
            loss.backward()
            agent.optimizer_step()

        if plot:
            plt.plot(history)
            plt.show()

    def train(self, agent, num_episodes, batch_size, train_steps=1, eval_episodes=0, td_steps=1, plot=False):
        episode_reward = []
        evaluation_episode = []
        evaluation_reward = []
        fig = plt.figure() if plot else None
        ax = fig.subplots() if fig is not None else None

        for ep in range(num_episodes):
            if ep % 10 == 0 and eval_episodes > 0:
                agent.eval()
                evaluation_episode.append(ep)
                evaluation_reward.append(self.eval(agent, eval_episodes))
                agent.train()

            done = False
            curr_reward = 0
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

                # self.buffer.append(state, policy_action, reward, next_state)
                self.buffer.append(state, policy_action, reward, episode_terminated=done)
                state = next_state

                if len(self.buffer) > batch_size:
                    for _ in range(train_steps):
                        # agent.step(self.buffer.sample(batch_size=batch_size))
                        agent.step(self.buffer.sample(batch_size=batch_size, trajectory_length=td_steps))

            episode_reward.append(curr_reward)

            if plot:
                ax.cla()
                ax.plot(episode_reward)
                if eval_episodes > 0:
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
