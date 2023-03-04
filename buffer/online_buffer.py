from buffer.abstract_buffer import AbstractBuffer
from buffer.transitions import ActionTransition

class OnlineEpisodeBuffer(AbstractBuffer):
    def __init__(self):
        super().__init__()
        self.episode = []
        self.buffer = []

    def __len__(self):
        return sum(len(episode) for episode in self.buffer)

    def append(self, action, reward, episode_terminated=False):
        self.episode.append(ActionTransition(action, reward))
        if episode_terminated:
            self.buffer.append(self.episode)
            self.episode = []

    def sample(self):
        return self.buffer

    def clear(self):
        self.buffer = []

if __name__ == '__main__':
    b = OnlineEpisodeBuffer()

    b.append([1, 2], 2)
    b.append([2, 3], 3)
    b.append([3, 4], 4)

    print(b.buffer)
    print(b.sample())