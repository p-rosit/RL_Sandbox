import torch
from torch import nn
from buffer.transitions import Transition
from core.abstract_agent import AbstractAgent
from core.network_wrappers import SoftUpdateModel

class DoubleQLearningAgent(AbstractAgent):
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
        batch_experiences = Transition(*zip(*experiences))

        states = torch.cat(batch_experiences.state, dim=0)
        actions = torch.cat(batch_experiences.action, dim=0)
        rewards = torch.cat(batch_experiences.reward, dim=0)

        non_final_mask = torch.tensor([next_state is not None for next_state in batch_experiences.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([next_state for next_state in batch_experiences.next_state if next_state is not None])

        experiences = (states, actions, rewards, non_final_next_states, non_final_mask)

        loss = torch.tensor([0], dtype=torch.float64)

        loss += self._compute_loss(self.policy_network_1, self.target_network_2, experiences)
        loss += self._compute_loss(self.policy_network_2, self.target_network_1, experiences)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.parameters(), self.max_grad)
        self.optimizer.step()

        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)

class ModifiedDoubleQLearningAgent(AbstractAgent):
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
        return self.policy_network(state).argmax(dim=1).view(1, 1)

    def parameters(self):
        return self.policy_network.parameters()

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
        batch_experiences = Transition(*zip(*experiences))

        states = torch.cat(batch_experiences.state, dim=0)
        actions = torch.cat(batch_experiences.action, dim=0)
        rewards = torch.cat(batch_experiences.reward, dim=0)

        non_final_mask = torch.tensor([next_state is not None for next_state in batch_experiences.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([next_state for next_state in batch_experiences.next_state if next_state is not None])

        experiences = (states, actions, rewards, non_final_next_states, non_final_mask)

        loss = self._compute_loss(self.policy_network, self.target_network, experiences)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.parameters(), self.max_grad)
        self.optimizer.step()

        self.target_network.update(self.policy_network)

class ClippedDoubleQLearning(AbstractAgent):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99, tau=0.005, max_grad=100):
        pass