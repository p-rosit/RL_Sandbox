from abstract_actor import AbtractActor

class RandomActor(AbtractActor):
    def __init__(self, env):
        self.env = env

    def sample(self, state):
        return self.env.action_space.sample()

    def step(self, *args):
        pass
