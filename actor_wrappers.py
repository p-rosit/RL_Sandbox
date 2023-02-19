import numpy as np
from abstract_actor import AbtractActor

rng = np.random.default_rng()

class Discrete2Continuous(AbtractActor):
    def __init__(self, actor, remap):
        super().__init__()
        self.actor = actor
        self.remap = remap
        self.reverse_map = {val: ind for ind, val in enumerate(self.remap)}

    def sample(self, state):
        return self.remap[self.actor.sample(state)]

    def set_optimizer(self, optimizer):
        self.actor.set_optimizer(optimizer)

    def set_criterion(self, criterion):
        self.actor.set_criterion(criterion)

    def parameters(self):
        return self.actor.parameters()

    def step(self, continuous_experiences):
        discrete_experiences = []
        for state, continuous_action, reward, next_state in continuous_experiences:
            discrete_experiences.append((state, self.reverse_map[continuous_action], reward, next_state))
        self.actor.step(discrete_experiences)

class MultiActor(AbtractActor):
    def __init__(self, *actors, p=[]):
        self.actors = actors
        self.p = p
        if len(self.actors) != len(self.p):
            raise ValueError('Not same length')

    def sample(self, state):
        actor = rng.choice(self.actors, p=self.p)
        return actor.sample(state)

    def set_optimizer(self, optimizer):
        raise AttributeError("Class %s does not require an optimizer." % self.__class__.__name__)

    def set_criterion(self, criterion):
        raise AttributeError("Class %s does not require a criterion." % self.__class__.__name__)

    def parameters(self):
        raise AttributeError("Class %s does not have any parameters." % self.__class__.__name__)

    def step(self, *args):
        for actor in self.actors:
            actor.step(*args)
