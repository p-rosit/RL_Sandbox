import numpy as np
# import torch
from buffer.transitions import Experience
from buffer.abstract_buffer import AbstractBuffer

rng = np.random.default_rng()

class ReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=100, default_prob=10.0):
        super().__init__(max_size=max_size)
        self.ind = 0
        self.buffer = []
        self.weights = []
        self.default_prob = default_prob

        self.episode_inds = []
        self.inds = []

        self.episode = []

    def __len__(self):
        return sum(episode.state.shape[0] for episode in self.buffer)

    def normalize(self):
        episode_sum = [episode.sum() for episode in self.weights]
        total_sum = sum(episode_sum)

        episode_weights = [weight / total_sum for weight in episode_sum]
        normalized_weights = [episode / ep_sum for ep_sum, episode in zip(episode_sum, self.weights)]
        return episode_weights, normalized_weights

    def append(self, state, action, reward, episode_terminated=False):
        experience = Experience(state.numpy(), action.numpy(), reward.numpy())
        self.episode.append(experience)

        if episode_terminated:
            states, actions, rewards = zip(*self.episode)
            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            episode = Experience(states, actions, rewards)

            self.buffer.append(episode)
            self.weights.append(np.full(len(self.episode), self.default_prob))

            self.episode = []
            while len(self) > self.max_size and len(self.buffer) > 1:
                self.buffer.pop(0)
                self.weights.pop(0)

    def sample(self, batch_size=1, trajectory_length=1):
        episode_weights, normalized_weights = self.normalize()

        self.episode_inds = rng.choice(len(self.buffer), batch_size, p=episode_weights)
        self.inds = []

        trajectories = []
        for episode_ind in self.episode_inds:
            episode = self.buffer[episode_ind]
            ind = rng.choice(episode.state.shape[0], p=normalized_weights[episode_ind])
            self.inds.append(ind)

            episode_states, episode_actions, episode_rewards = episode

            if episode_states.shape[0] > ind + trajectory_length:
                states = episode_states[ind:(ind+trajectory_length+1)]
            else:
                states = episode_states[ind:(ind+trajectory_length)]
            actions = episode_actions[ind:min(episode_actions.shape[0], ind+trajectory_length)]
            rewards = episode_rewards[ind:min(episode_rewards.shape[0], ind+trajectory_length)]

            trajectories.append(Experience(states, actions, rewards))

        return trajectories

    def update(self, sample_errors):
        for i, (episode_ind, ind) in enumerate(zip(self.episode_inds, self.inds)):
            self.weights[episode_ind][ind] = sample_errors[i]

    def all_episodes(self):
        return self.buffer

    def clear(self):
        self.buffer = []
        self.weights = []
        self.episode = []

if __name__ == '__main__':
    b = ReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3)
    b.append([2, 3], 3, 4)
    b.append([3, 4], 4, 5)

    print(b.buffer)
    print(b.sample(2))
