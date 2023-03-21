class AbstractBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size

    def __len__(self):
        raise NotImplementedError

    def _normalize_weights(self):
        raise NotImplementedError

    def update_weights(self, sample_errors):
        raise NotImplementedError

    def append(self, state, action, reward, episode_terminated=False):
        raise NotImplementedError

    def sample(self, batch_size=1, trajectory_length=1):
        raise NotImplementedError

    def all_episodes(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
