from abstract_actor import AbstractActor

class RandomActor(AbstractActor):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def sample(self, state):
        return self.env.action_space.sample()

    def set_optimizer(self, optimizer):
        raise AttributeError("Class %s does not require an optimizer." % self.__class__.__name__)

    def set_criterion(self, criterion):
        raise AttributeError("Class %s does not require a criterion." % self.__class__.__name__)

    def parameters(self):
        raise AttributeError("Class %s does not have any parameters." % self.__class__.__name__)

    def step(self, *args):
        pass
