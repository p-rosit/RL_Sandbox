from abstract_actor import AbtractActor

class RandomActor(AbtractActor):
    def __init__(self, env):
        self.env = env

    def sample(self):
        return self.env.action_space.sample()

    def train(self, *args):
        pass
