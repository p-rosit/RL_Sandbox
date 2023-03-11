import torch
from core.agents.abstract_agent import AbstractAgent

class AbstractActorCritic(AbstractAgent):
    def __init__(self):
        super().__init__()

    def sample(self, state):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def loss(self, experiences):
        raise NotImplementedError
