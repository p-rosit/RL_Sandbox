import torch
from torch import nn
from buffer.transitions import Transition, batch_transitions
from agents.off_policy.deep_q_learning.abstract_q_learning_agent import AbstractQLearningAgent, AbstractDoubleQLearningAgent
from core.network_wrappers import SoftUpdateModel

from core.abstract_agent import AbstractAgent

def policy_target_pair(layer_sizes, tau):
    network = []
    for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        network.append(nn.Linear(size_in, size_out))
        network.append(nn.ReLU())
    network.pop()

    policy_network = nn.Sequential(*network)
    target_network = SoftUpdateModel(policy_network, tau=tau)

    return policy_network, target_network

class DoubleQLearningAgent(AbstractDoubleQLearningAgent):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99, tau=0.005, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)
        layers = (input_size, *layer_sizes, output_size)

        self.policy_network_1, self.target_network_1 = policy_target_pair(layers, tau)
        self.policy_network_2, self.target_network_2 = policy_target_pair(layers, tau)

    def _compute_loss(self, policy_network, target_network, states, actions, rewards, non_final_next_states, non_final_mask):
        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            estimated_next_actions = policy_network(non_final_next_states).argmax(dim=1).view(-1, 1)

            estimated_next_values = target_network(non_final_next_states)
            estimated_next_action_values = estimated_next_values.gather(1, estimated_next_actions).squeeze()

        bellman_action_values = torch.clone(rewards)
        bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values

        return self.criterion(estimated_action_values, bellman_action_values)

    def step(self, experiences):
        super().step(experiences)
        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)

class ModifiedDoubleQLearningAgent(AbstractQLearningAgent):
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

        estimated_next_action_values = torch.zeros_like(rewards)
        with torch.no_grad():
            estimated_next_actions = policy_network(non_final_next_states).argmax(dim=1).view(-1, 1)
            estimated_next_values = target_network(non_final_next_states)
            estimated_next_action_values[non_final_mask] = estimated_next_values.gather(1, estimated_next_actions).squeeze()

        bellman_action_values = rewards + self.discount * estimated_next_action_values

        return self.criterion(estimated_action_values, bellman_action_values)

    def step(self, experiences):
        super().step(experiences)
        self.target_network.update(self.policy_network)

class ClippedDoubleQLearning(AbstractAgent):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99, tau=0.005, max_grad=100):
        super().__init__()
        layers = (input_size, *layer_sizes, output_size)

        network_1 = []
        network_2 = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            network_1.append(nn.Linear(size_in, size_out))
            network_1.append(nn.ReLU())
            network_2.append(nn.Linear(size_in, size_out))
            network_2.append(nn.ReLU())
        network_1.pop()
        network_2.pop()

        self.discount = discount
        self.policy_network_1 = nn.Sequential(*network_1)
        self.policy_network_2 = nn.Sequential(*network_2)
        self.target_network_1 = SoftUpdateModel(self.policy_network_1, tau=tau)
        self.target_network_2 = SoftUpdateModel(self.policy_network_2, tau=tau)
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

    def step(self, experiences):
        batch_experiences = Transition(*zip(*experiences))

        states = torch.cat(batch_experiences.state, dim=0)
        actions = torch.cat(batch_experiences.action, dim=0)
        rewards = torch.cat(batch_experiences.reward, dim=0)

        non_final_mask = torch.tensor([next_state is not None for next_state in batch_experiences.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([next_state for next_state in batch_experiences.next_state if next_state is not None])

        estimated_action_values_1 = self.policy_network_1(states).gather(1, actions).squeeze()
        estimated_action_values_2 = self.policy_network_2(states).gather(1, actions).squeeze()

        estimated_next_action_values = torch.zeros_like(rewards, dtype=torch.float64)
        with torch.no_grad():
            estimated_next_values_1, _ = self.target_network_1(non_final_next_states).max(dim=1)
            estimated_next_values_2, _ = self.target_network_2(non_final_next_states).max(dim=1)
            estimated_next_action_values[non_final_mask] = torch.min(estimated_next_values_1, estimated_next_values_2)

        bellman_action_values = rewards + self.discount * estimated_next_action_values

        loss_1 = self.criterion(estimated_action_values_1, bellman_action_values)
        loss_2 = self.criterion(estimated_action_values_2, bellman_action_values)

        self.optimizer.zero_grad()
        (loss_1 + loss_2).backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.max_grad)
        self.optimizer.step()

        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)
