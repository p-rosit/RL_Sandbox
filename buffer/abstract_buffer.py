class AbstractBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size

    def __len__(self):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
