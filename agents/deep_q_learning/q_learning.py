import torch
from torch import nn
from buffer.transitions import Transition
from core.abstract_agent import AbstractAgent
from core.network_wrappers import SoftUpdateModel

class DenseQLearningAgent(AbstractAgent):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99, tau=0.005, max_grad=100):
        super().__init__()
        layers = (input_size, *layer_sizes, output_size)

        network = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            network.append(nn.Linear(size_in, size_out))
            network.append(nn.ReLU())
        network.pop()

        self.discount = discount
        self.policy_network = nn.Sequential(*network)
        self.target_network = SoftUpdateModel(self.policy_network, tau=tau)
        self.max_grad = max_grad

    def sample(self, state):
        values = self.policy_network(state)
        return torch.argmax(values, dim=1).view(1, 1)

    def parameters(self):
        return self.policy_network.parameters()

    def step(self, experiences):
        batch_experiences = Transition(*zip(*experiences))

        states = torch.cat(batch_experiences.state, dim=0)
        actions = torch.cat(batch_experiences.action, dim=0)
        rewards = torch.cat(batch_experiences.reward, dim=0)

        non_final_mask = torch.tensor([next_state is not None for next_state in batch_experiences.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([next_state for next_state in batch_experiences.next_state if next_state is not None])

        estimated_next_action_values = torch.zeros_like(rewards)
        with torch.no_grad():
            estimated_next_action_values[non_final_mask], _ = self.target_network(non_final_next_states).max(dim=1)

        estimated_action_values = self.policy_network(states).gather(1, actions).squeeze()
        bellman_action_values = rewards + self.discount * estimated_next_action_values

        loss = self.criterion(estimated_action_values, bellman_action_values)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), self.max_grad)
        self.optimizer.step()

        self.target_network.update(self.policy_network)
