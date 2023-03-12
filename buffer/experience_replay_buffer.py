import numpy as np
import torch
from buffer.transitions import Transition, ActionTransition
from buffer.abstract_buffer import AbstractBuffer

rng = np.random.default_rng()

class ReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=100, default_prob=100):
        super().__init__(max_size=max_size)
        self.ind = 0
        self.buffer = []
        self.weights = []
        self.default_prob = default_prob

        self.episode_inds = []
        self.inds = []

        self.episode = []

    def __len__(self):
        return sum(len(episode) for episode in self.buffer)

    def normalize(self):
        episode_sum = [sum(episode) for episode in self.weights]
        total_sum = sum(episode_sum)

        episode_weights = [weight / total_sum for weight in episode_sum]
        normalized_weights = [[weight / ep_sum for weight in episode] for ep_sum, episode in zip(episode_sum, self.weights)]
        return episode_weights, normalized_weights

    def append(self, state, action, reward, episode_terminated=False):
        experience = ActionTransition(state, action, reward)
        self.episode.append(experience)

        if episode_terminated:
            self.buffer.append(self.episode)
            self.weights.append([self.default_prob for _ in self.episode])

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
            ind = rng.choice(len(self.buffer[episode_ind]), p=normalized_weights[episode_ind])
            self.inds.append(ind)

            experiences = self.buffer[episode_ind][ind:min(len(self.buffer[episode_ind]), ind+trajectory_length)]
            states, actions, rewards = zip(*experiences)

            if len(self.buffer[episode_ind]) > ind + trajectory_length:
                states = (*states, self.buffer[episode_ind][ind + trajectory_length].state)

            trajectories.append(ActionTransition(states, actions, rewards))

        return trajectories

    def update(self, sample_errors):
        for episode_ind, ind, sample_error in zip(self.episode_inds, self.inds, sample_errors):
            self.weights[episode_ind][ind] = sample_error.item()

    def all_episodes(self):
        episodes = []
        for episode in self.buffer:
            states, actions, rewards = zip(*episode)
            episodes.append(ActionTransition(states, actions, rewards))

        return episodes

    def clear(self):
        self.buffer = []
        self.weights = []
        self.episode = []

if __name__ == '__main__':
    b = ReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3, [4, 5])
    b.append([2, 3], 3, 4, [5, 6])
    b.append([3, 4], 4, 5, [6, 7])

    print(b.buffer)
    print(b.sample(2))
