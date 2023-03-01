import numpy as np
from buffer.transitions import Transition

rng = np.random.default_rng()

class EpisodeBuffer:
    def __init__(self, max_size=100):
        self.ind = 0
        self.max_size = max_size
        self.buffer = []
        self.episode = []

    def __len__(self):
        return sum(len(episode) for episode in self.buffer)

    def finish_episode(self):
        if len(self.buffer) < self.max_size:
            self.buffer.append(self.episode)
        else:
            self.buffer[self.ind] = self.episode

        self.episode = []
        self.ind = (self.ind + 1) % self.max_size

    def append(self, state, action, reward, next_state):
        self.episode.append(Transition(state, action, reward, next_state))

    def sample(self, batch_size=1):
        # sample episode or sample trajectories from episodes?
        return [self.buffer[ind] for ind in rng.choice(len(self.buffer), batch_size)]

if __name__ == '__main__':
    b = EpisodeBuffer(max_size=2)

    b.append([1, 2], 2, 3, [4, 5])
    b.append([2, 3], 3, 4, [5, 6])
    b.finish_episode()
    b.append([3, 4], 4, 5, [6, 7])

    print(b.buffer)
    print(b.sample(2))
