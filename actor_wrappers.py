import numpy as np
from abstract_actor import AbstractActor
from copy import deepcopy

rng = np.random.default_rng()

class Discrete2Continuous(AbstractActor):
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

class MultiActor(AbstractActor):
    def __init__(self, *actors, p=None):
        super().__init__()
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

class AnnealActor(AbstractActor):
    def __init__(self, actor, replacement_actor, start_steps=1000, eps_start=0.9, eps_end=0.05, decay_steps=1000):
        super().__init__()
        self.actor = actor
        self.replacement_actor = replacement_actor
        self.start_steps = start_steps
        self.decay_steps = decay_steps
        self.eps_start = eps_start
        self.eps_end = eps_end

        self.curr_step = 0

    def reset_annealing(self):
        self.curr_step = 0

    def sample(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * (self.curr_step - self.start_steps) / self.decay_steps)
        self.curr_step += 1

        if self.curr_step <= self.start_steps or rng.random() < eps:
            return self.replacement_actor.sample(state)
        else:
            return self.actor.sample(state)

    def set_optimizer(self, optimizer):
        self.actor.set_optimizer(optimizer)

    def set_criterion(self, criterion):
        self.actor.set_criterion(criterion)

    def parameters(self):
        return self.actor.parameters()

    def step(self, *args):
        self.actor.step(*args)

class SoftUpdateModel:
    def __init__(self, actor, tau=0.005):
        self.actor = deepcopy(actor)
        self.tau = tau

    def __call__(self, x):
        return self.actor(x)

    def update(self, actor):
        policy_state_dict = actor.state_dict()
        target_state_dict = self.actor.state_dict()
        try:
            for key in target_state_dict:
                target_state_dict[key] = self.tau * policy_state_dict[key] + (1 - self.tau) * target_state_dict[key]
            self.actor.load_state_dict(target_state_dict)
        except KeyError:
            raise KeyError('Saved model and new model do not have the same parameters.')
