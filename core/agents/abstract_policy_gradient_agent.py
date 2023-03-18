import torch
from torch.distributions import Categorical
from buffer.transitions import batch_trajectories, batch_episodes
from core.agents.abstract_agent import AbstractAgent

class AbstractPolicyGradientAgent(AbstractAgent):
    def __init__(self, policy_network, discount=0.99):
        super().__init__(discount=discount)
        self.policy_network = policy_network

    def train(self):
        self.policy_network.train()
        super().train()
    
    def eval(self):
        self.policy_network.eval()
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

    def _compute_loss(self, states, actions, reward):
        raise NotImplementedError

    def _loss(self, states, actions, rewards):
        return self._compute_loss(states, actions, rewards)

    def loss(self, experiences):
        states, actions, rewards = batch_episodes(experiences)
        return self._loss(states, actions, rewards)
