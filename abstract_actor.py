import torch
import numpy as np

class AbstractActor:
    def __init__(self):
        self.optimizer = None
        self.criterion = None

    def sample(self, state):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def parameters(self):
        raise NotImplementedError

    def _step(self, experiences):
        states = torch.tensor(np.array([state for state, _, _, _ in experiences]))
        actions = torch.tensor(np.array([action for _, action, _, _ in experiences]))
        rewards = torch.tensor(np.array([reward for _, _, reward, _ in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([next_state for _, _, _, next_state in experiences]))
        return states, actions, rewards, next_states

    def step(self, experiences):
        raise NotImplementedError
