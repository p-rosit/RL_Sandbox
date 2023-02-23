import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
rng = np.random.default_rng()

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.ind = 0
        self.max_size = max_size
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

if __name__ == '__main__':
    b = ReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3, [4, 5])
    b.append([2, 3], 3, 4, [5, 6])
    b.append([3, 4], 4, 5, [6, 7])

    print(b.buffer)
    print(b.sample(2))
