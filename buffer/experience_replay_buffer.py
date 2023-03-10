import numpy as np
from buffer.transitions import Transition, ActionTransition
from buffer.abstract_buffer import AbstractBuffer

rng = np.random.default_rng()

class ReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=1000):
        super().__init__(max_size=max_size)
        self.ind = 0
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state):
        experience = Transition(state, action, reward, next_state)

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ind] = experience

        self.ind = (self.ind + 1) % self.max_size

    def sample(self, batch_size=1):
        return [self.buffer[ind] for ind in rng.choice(len(self.buffer), batch_size)]

    def clear(self):
        self.buffer = []

class ModReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=100):
        super().__init__(max_size=max_size)
        self.ind = 0
        self.buffer = []
        self.episode = []
        self.episode_inds = []
        self.inds = []

    def __len__(self):
        return sum(len(episode) for episode in self.buffer)

    def append(self, state, action, reward, episode_terminated=False):
        experience = ActionTransition(state, action, reward)
        self.episode.append(experience)

        if episode_terminated:
            self.buffer.append(self.episode)
            self.episode = []
            while len(self) > self.max_size and len(self.buffer) > 1:
                self.buffer.pop(0)

    def sample(self, batch_size=1, trajectory_length=1):
        self.episode_inds = rng.choice(len(self.buffer), batch_size)
        self.inds = []

        trajectories = []
        for episode_ind in self.episode_inds:
            ind = rng.choice(len(self.buffer[episode_ind]))
            self.inds.append(ind)

            experiences = self.buffer[episode_ind][ind:ind+trajectory_length]
            states, actions, rewards = zip(*experiences)

            if len(self.buffer[episode_ind]) > ind + trajectory_length:
                states = (*states, self.buffer[episode_ind][ind + trajectory_length].state)
            else:
                states = (*states, None)

            trajectories.append(ActionTransition(states, actions, rewards))

        return trajectories

    def update(self, sample_error):
        pass

    def all_episodes(self):
        return self.buffer

    def clear(self):
        self.buffer = []
        self.episode = []

if __name__ == '__main__':
    b = ReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3, [4, 5])
    b.append([2, 3], 3, 4, [5, 6])
    b.append([3, 4], 4, 5, [6, 7])

    print(b.buffer)
    print(b.sample(2))
