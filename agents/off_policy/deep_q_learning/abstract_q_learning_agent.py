import torch
from buffer.transitions import batch_transitions
from core.abstract_agent import AbstractAgent

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

    def _compute_loss(self, policy_network, target_network, experiences):
        raise NotImplementedError

    def step(self, experiences):
        experiences = batch_transitions(experiences)
        loss = self._compute_loss(self.policy_network, self.target_network, experiences)
        super()._step(loss)

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

    def _compute_loss(self, policy_network, target_network, states, actions, rewards, non_final_next_states, non_final_mask):
        raise NotImplementedError

    def step(self, experiences):
        experiences = batch_transitions(experiences)

        loss_1 = self._compute_loss(self.policy_network_1, self.target_network_2, *experiences)
        loss_2 = self._compute_loss(self.policy_network_2, self.target_network_1, *experiences)

        super()._step(loss_1 + loss_2)

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
        network_step(loss, self.optimizer, self.parameters(), self.max_grad)
