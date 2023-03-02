import torch
from buffer.transitions import batch_transitions
from core.agents.abstract_q_learning_agent import AbstractMultiQlearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class MultiQLearningAgent(AbstractMultiQlearningAgent):
    def __init__(self, *networks, discount=0.99, tau=0.005, max_grad=100, policy_train=False):
        super().__init__(discount=discount, max_grad=max_grad)
        self.estimated_next_action_values = None

        self.policy_networks = []
        self.target_networks = []

        for policy_network in networks:
            self.policy_networks.append(policy_network)
            self.target_networks.append(SoftUpdateModel(policy_network, tau=tau))

        self.policy_train = policy_train

    def _compute_loss(self, ind, states, actions, rewards, non_final_next_states, non_final_mask):
        estimated_action_values = self.policy_networks[ind](states).gather(1, actions).squeeze()

        with torch.no_grad():
            if self.policy_train:
                estimated_next_actions = self.policy_networks[ind](non_final_next_states).argmax(dim=1)
            else:
                estimated_next_actions = self.target_networks[ind](non_final_next_states).argmax(dim=1)

            estimated_next_actions = estimated_next_actions.view(1, -1, 1).repeat(len(self.target_networks), 1, 1)

            estimated_next_action_values = self.estimated_next_action_values[:ind].gather(2, estimated_next_actions[:ind]).sum(dim=0)
            estimated_next_action_values += self.estimated_next_action_values[ind+1:].gather(2, estimated_next_actions[ind+1:]).sum(dim=0)

            estimated_next_action_values /= len(self.target_networks) - 1
            estimated_next_action_values = estimated_next_action_values.view(-1)

        bellman_action_values = rewards.clone()
        bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values

        return self.criterion(estimated_action_values, bellman_action_values)

    def step(self, experiences):
        _, _, _, non_final_next_states, _ = batch_transitions(experiences)

        estimated_next_action_values = []
        with torch.no_grad():
            for target_network in self.target_networks:
                vals = target_network(non_final_next_states).unsqueeze(0)
                estimated_next_action_values.append(vals)
        self.estimated_next_action_values = torch.cat(estimated_next_action_values, dim=0)

        super().step(experiences)

        for policy_network, target_network in zip(self.policy_networks, self.target_networks):
            target_network.update(policy_network)
