import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class reinforce(AbstractAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _compute_loss(self):
        pass

    def _step(self, experiences):
        pass
