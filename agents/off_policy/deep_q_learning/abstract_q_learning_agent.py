import torch
from core.abstract_agent import AbstractAgent

def network_step(loss, parameters, optimizer, max_grad):
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(parameters, max_grad)
    optimizer.step()

class AbstractQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__()
        self.policy_network = None
        self.target_network = None
        self.discount = discount
        self.max_grad = max_grad

    def sample(self, state):
        return self.policy_network(state).argmax(dim=1).view(1, 1)

    def parameters(self):
        return self.policy_network.parameters()

    def step(self, loss):
        network_step(loss, self.parameters(), self.optimizer, self.max_grad)

class AbstractDoubleQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__()
        self.policy_network_1 = None
        self.policy_network_2 = None
        self.target_network_1 = None
        self.target_network_2 = None
        self.discount = discount
        self.max_grad = max_grad

    def sample(self, state):
        if self.training:
            if torch.rand(1) < 0.5:
                return self.policy_network_1(state).argmax(dim=1).view(1, 1)
            else:
                return self.policy_network_2(state).argmax(dim=1).view(1, 1)
        else:
            value_1, action_1 = self.policy_network_1(state).max(dim=1)
            value_2, action_2 = self.policy_network_2(state).max(dim=1)
            if value_1 > value_2:
                return action_1
            else:
                return action_2

    def parameters(self):
        return *self.policy_network_1.parameters(), *self.policy_network_2.parameters()

    def step(self, loss):
        network_step(loss, self.parameters(), self.optimizer, self.max_grad)

class AbstractMultiQlearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__()
        self.policy_networks = None
        self.target_networks = None
        self.discount = discount
        self.max_grad = max_grad

    def sample(self, state):
        pass

    def parameters(self):
        for policy_network in self.policy_networks:
            for param in policy_network.parameters():
                yield param

    def step(self, loss):
        network_step(loss, self.parameters(), self.optimizer, self.max_grad)
