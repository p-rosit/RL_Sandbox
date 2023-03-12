import torch
from core.agents.abstract_q_learning_agent import AbstractMultiQlearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class MultiQLearningAgent(AbstractMultiQlearningAgent):
    def __init__(self, *networks, discount=0.99, tau=0.005, policy_train=False):
        super().__init__(discount=discount)
        self.estimated_next_action_values = None

        self.policy_networks = []
        self.target_networks = []

        for policy_network in networks:
            self.policy_networks.append(policy_network)
            self.target_networks.append(SoftUpdateModel(policy_network, tau=tau))

        self.policy_train = policy_train

    def _compute_loss(self, ind, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = self.policy_networks[ind](states[0]).gather(1, actions[0].reshape(-1, 1)).squeeze()

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            if self.policy_train:
                estimated_next_actions = self.policy_networks[ind](next_states).argmax(dim=1)
            else:
                estimated_next_actions = self.target_networks[ind](next_states).argmax(dim=1)

            estimated_next_actions = estimated_next_actions.view(1, -1, 1).repeat(len(self.target_networks), 1, 1)

            estimated_next_action_values = self.estimated_next_action_values[:ind].gather(2, estimated_next_actions[:ind]).sum(dim=0)
            estimated_next_action_values += self.estimated_next_action_values[ind+1:].gather(2, estimated_next_actions[ind+1:]).sum(dim=0)

            estimated_next_action_values /= len(self.target_networks) - 1
            estimated_next_action_values = estimated_next_action_values.view(-1)

        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = self.policy_networks[ind].intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error

    def _loss(self, states, actions, rewards, masks):
        estimated_next_action_values = []
        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            for target_network in self.target_networks:
                vals = target_network(next_states).unsqueeze(0)
                estimated_next_action_values.append(vals)
        self.estimated_next_action_values = torch.cat(estimated_next_action_values, dim=0)

        loss, td_error = super()._loss(states, actions, rewards, masks)
        self.estimated_next_action_values = None
        return loss, td_error

class ClippedMultiQLearningAgent(AbstractMultiQlearningAgent):

    def __init__(self, *networks, discount=0.99, tau=0.005, policy_train=False):
        super().__init__(discount=discount)
        self.estimated_next_action_values = None

        self.policy_networks = []
        self.target_networks = []

        for policy_network in networks:
            self.policy_networks.append(policy_network)
            self.target_networks.append(SoftUpdateModel(policy_network, tau=tau))

        self.policy_train = policy_train

    def _compute_loss(self, ind, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = self.policy_networks[ind](states[0]).gather(1, actions[0].reshape(-1, 1)).squeeze()

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            if self.policy_train:
                estimated_next_actions = self.policy_networks[ind](next_states).argmax(dim=1)
            else:
                estimated_next_actions = self.target_networks[ind](next_states).argmax(dim=1)

            estimated_next_actions = estimated_next_actions.view(1, -1, 1).repeat(len(self.target_networks), 1, 1)

            if ind != 0:
                vals_1, _ = self.estimated_next_action_values[:ind].gather(2, estimated_next_actions[:ind]).min(dim=0)
            else:
                vals_1 = torch.full((next_states.size(0), 1), torch.inf)
            if ind != len(self.policy_networks) - 1:
                vals_2, _ = self.estimated_next_action_values[ind+1:].gather(2, estimated_next_actions[ind+1:]).min(dim=0)
            else:
                vals_2 = torch.full((next_states.size(0), 1), torch.inf)

            estimated_next_action_values = torch.min(vals_1, vals_2)
            estimated_next_action_values = estimated_next_action_values.view(-1)

        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = self.policy_networks[ind].intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error

    def _loss(self, states, actions, rewards, masks):
        estimated_next_action_values = []
        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            for target_network in self.target_networks:
                vals = target_network(next_states).unsqueeze(0)
                estimated_next_action_values.append(vals)
        self.estimated_next_action_values = torch.cat(estimated_next_action_values, dim=0)

        loss, td_error = super()._loss(states, actions, rewards, masks)
        self.estimated_next_action_values = None
        return loss, td_error
