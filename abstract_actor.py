class AbtractActor:
    def sample(self, state):
        raise NotImplementedError

    def step(self, experiences):
        raise NotImplementedError
