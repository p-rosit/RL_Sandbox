import numpy as np
from buffer.transitions import Transition
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

if __name__ == '__main__':
    b = ReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3, [4, 5])
    b.append([2, 3], 3, 4, [5, 6])
    b.append([3, 4], 4, 5, [6, 7])

    print(b.buffer)
    print(b.sample(2))
