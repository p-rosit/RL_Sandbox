import torch
from torch import nn
from buffer.transitions import batch_trajectories
from core.agents.abstract_agent import AbstractAgent

class AbstractQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99):
        super().__init__(discount=discount)
        self.criterion = nn.SmoothL1Loss()
        self.policy_network = None
        self.target_network = None

    def train(self):
        self.policy_network.train()
        self.target_network.train()
        super().train()

    def eval(self):
        self.policy_network.eval()
        self.target_network.eval()
        super().eval()

    def sample(self, state):
        policy_action, env_action = self.policy_network.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return self.policy_network.parameters()

    def _pretrain_loss(self, states, actions, rewards, masks):
        return self.policy_network.pretrain_loss(states, actions, rewards, masks)

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _compute_loss(self, states, actions, rewards, masks):
        raise NotImplementedError

    def _loss(self, states, actions, rewards, masks):
        return self._compute_loss(states, actions, rewards, masks)

    def loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._loss(*batch_experiences)

    def update_target(self):
        self.target_network.update(self.policy_network)

class AbstractDoubleQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99):
        super().__init__(discount=discount)
        self.criterion = nn.SmoothL1Loss()
        self.policy_network_1 = None
        self.policy_network_2 = None
        self.target_network_1 = None
        self.target_network_2 = None

    def train(self):
        self.policy_network_1.train()
        self.policy_network_2.train()
        self.target_network_1.train()
        self.target_network_2.train()
        super().train()

    def eval(self):
        self.policy_network_1.eval()
        self.policy_network_2.eval()
        self.target_network_1.eval()
        self.target_network_2.eval()
        super().eval()

    def sample(self, state):
        if self.training:
            if torch.rand(1) < 0.5:
                policy_action, env_action = self.policy_network_1.action(state)
            else:
                policy_action, env_action = self.policy_network_2.action(state)
        else:
            value_1, action_1 = self.policy_network_1.action_value(state)
            value_2, action_2 = self.policy_network_2.action_value(state)
            if value_1 > value_2:
                policy_action = action_1
                env_action = action_1.item()
            else:
                policy_action = action_2
                env_action = action_1.item()

        return policy_action.view(-1, 1), env_action

    def parameters(self):
        for param in self.policy_network_1.parameters():
            yield param
        for param in self.policy_network_2.parameters():
            yield param

    def _pretrain_loss(self, states, actions, rewards, masks):
        loss_1 = self.policy_network_1.pretrain_loss(states, actions, rewards, masks)
        loss_2 = self.policy_network_2.pretrain_loss(states, actions, rewards, masks)
        return loss_1 + loss_2

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _loss(self, states, actions, rewards, masks):
        loss_1, td_error_1 = self._compute_loss(0, states, actions, rewards, masks)
        loss_2, td_error_2 = self._compute_loss(1, states, actions, rewards, masks)
        return loss_1 + loss_2, (td_error_1 + td_error_2) / 2

    def loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._loss(*batch_experiences)

    def update_target(self):
        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)

class AbstractMultiQlearningAgent(AbstractAgent):
    def __init__(self, discount=0.99):
        super().__init__(discount=discount)
        self.criterion = nn.SmoothL1Loss()
        self.policy_networks = None
        self.target_networks = None

    def train(self):
        for policy_network, target_network in zip(self.policy_networks, self.target_networks):
            policy_network.train()
            target_network.train()
        super().train()

    def eval(self):
        for policy_network, target_network in zip(self.policy_networks, self.target_networks):
            policy_network.eval()
            target_network.eval()
        super().eval()

    def sample(self, state):
        if self.training:
            ind = torch.randint(len(self.policy_networks), (1,))
            policy_action, env_action = self.policy_networks[ind].action(state)
        else:
            # fix vectorization
            actions = [policy_network.action_value(state) for policy_network in self.policy_networks]
            _, policy_action = max(actions, key=lambda x: x[0])

        return policy_action.view(-1, 1), policy_action.item()

    def parameters(self):
        for policy_network in self.policy_networks:
            for param in policy_network.parameters():
                yield param

    def _pretrain_loss(self, states, actions, rewards, masks):
        loss = torch.zeros(1)
        for policy_network in self.policy_networks:
            loss += policy_network.pretrain_loss(states, actions, rewards, masks)
        return loss

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _compute_loss(self, ind, states, actions, rewards, masks):
        raise NotImplementedError

    def _loss(self, states, actions, rewards, masks):
        loss = torch.zeros(1)
        td_error = torch.zeros(masks.size(1))
        for ind, policy_network in enumerate(self.policy_networks):
            l, td = self._compute_loss(ind, states, actions, rewards, masks)
            loss += l
            td_error += td
        return loss, td_error / masks.size(1)

    def loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._loss(*batch_experiences)

    def update_target(self):
        for policy_network, target_network in zip(self.policy_networks, self.target_networks):
            target_network.update(policy_network)
