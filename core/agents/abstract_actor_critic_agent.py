import torch
from core.agents.abstract_agent import AbstractAgent

class AbstractActorCritic(AbstractAgent):
    def __init__(self):
        super().__init__()

    def sample(self, state):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.max_grad)
        self.optimizer.step()

    def step(self, experiences):
        raise NotImplementedError
