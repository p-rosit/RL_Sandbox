import torch
from torch import nn
from agents.off_policy.deep_q_learning.abstract_q_learning_agent import AbstractQLearningAgent
from core.network_wrappers import SoftUpdateModel

class QLearningAgent(AbstractQLearningAgent):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99, tau=0.005, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)
        layers = (input_size, *layer_sizes, output_size)

        network = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            network.append(nn.Linear(size_in, size_out))
            network.append(nn.ReLU())
        network.pop()

        self.policy_network = nn.Sequential(*network)
        self.target_network = SoftUpdateModel(self.policy_network, tau=tau)

    def _compute_loss(self, policy_network, target_network, experiences):
        states, actions, rewards, non_final_next_states, non_final_mask = experiences
        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            estimated_next_action_values, _ = self.target_network(non_final_next_states).max(dim=1)
        bellman_action_values = rewards
        bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values

        return self.criterion(estimated_action_values, bellman_action_values)

    def step(self, experiences):
        super().step(experiences)
        self.target_network.update(self.policy_network)
