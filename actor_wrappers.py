import numpy as np
from abstract_actor import AbtractActor

rng = np.random.default_rng()

class Discrete2Continuous(AbtractActor):
    def __init__(self, actor, remap):
        self.actor = actor
        self.remap = remap

    def sample(self, state):
        return self.remap[self.actor.sample(state)]

    def step(self, *args):
        self.actor.step(*args)

class MultiActor(AbtractActor):
    def __init__(self, *actors, p=[]):
        self.actors = actors
        self.p = p
        if len(self.actors) != len(self.p):
            raise ValueError('Not same length')

    def sample(self, state):
        actor = rng.choice(self.actors, p=self.p)
        return actor.sample(state)

    def step(self, *args):
        for actor in self.actors:
            actor.step(*args)
