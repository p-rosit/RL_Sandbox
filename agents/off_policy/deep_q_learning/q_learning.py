import torch
from torch import nn
from buffer.transitions import batch_transitions
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

    def step(self, experiences):
        states, actions, rewards, non_final_next_states, non_final_mask = batch_transitions(experiences)

        estimated_next_action_values = torch.zeros_like(rewards)
        with torch.no_grad():
            estimated_next_action_values[non_final_mask], _ = self.target_network(non_final_next_states).max(dim=1)

        estimated_action_values = self.policy_network(states).gather(1, actions).squeeze()
        bellman_action_values = rewards + self.discount * estimated_next_action_values

        loss = self.criterion(estimated_action_values, bellman_action_values)

        super().step(loss)

        self.target_network.update(self.policy_network)
