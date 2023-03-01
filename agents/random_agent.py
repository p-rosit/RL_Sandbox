import torch
from core.abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def sample(self, state):
        env_action = self.env.action_space.sample()
        policy_action = torch.tensor(env_action, dtype=torch.long).view(1, 1)
        return policy_action, env_action

    def set_optimizer(self, optimizer):
        raise AttributeError("Class %s does not require an optimizer." % self.__class__.__name__)

    def set_criterion(self, criterion):
        raise AttributeError("Class %s does not require a criterion." % self.__class__.__name__)

    def parameters(self):
        raise AttributeError("Class %s does not have any parameters." % self.__class__.__name__)

    def step(self, *args):
        pass
