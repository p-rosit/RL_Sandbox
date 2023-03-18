import torch
from core.agents.abstract_q_learning_agent import AbstractQLearningAgent, AbstractDoubleQLearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class DoubleQLearningAgent(AbstractDoubleQLearningAgent):
    def __init__(self, policy_network_1, policy_network_2, discount=0.99, tau=0.005, policy_train=True):
        super().__init__(discount=discount)
        self.policy_train = policy_train
        self.policy_network_1 = policy_network_1
        self.target_network_1 = SoftUpdateModel(policy_network_1, tau=tau)

        self.policy_network_2 = policy_network_2
        self.target_network_2 = SoftUpdateModel(policy_network_2, tau=tau)

    def _compute_loss(self, ind, states, actions, rewards, masks):
        policy_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        target_network = self.target_network_2 if ind == 0 else self.target_network_1
        if self.policy_train:
            action_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        else:
            action_network = self.target_network_1 if ind == 0 else self.target_network_2

        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = policy_network(states[0]).gather(1, actions[0].reshape(-1, 1)).squeeze()

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            _, estimated_next_actions = action_network.action_value(next_states)
            estimated_next_action_values = target_network.value(next_states, estimated_next_actions).squeeze()

        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error

class ModifiedDoubleQLearningAgent(AbstractQLearningAgent):
    def __init__(self, policy_network, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.policy_network = policy_network
        self.target_network = SoftUpdateModel(policy_network, tau=tau)

    def _compute_loss(self, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = self.policy_network.value(states[0], actions[0]).squeeze()

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            _, estimated_next_actions = self.policy_network.action_value(next_states)
            estimated_next_action_values = self.target_network.value(next_states, estimated_next_actions).squeeze()

        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = self.policy_network.intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error

class ClippedDoubleQLearning(AbstractDoubleQLearningAgent):
    def __init__(self, policy_network_1, policy_network_2, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.bellman_action_values = None

        self.policy_network_1 = policy_network_1
        self.target_network_1 = SoftUpdateModel(policy_network_1, tau=tau)

        self.policy_network_2 = policy_network_2
        self.target_network_2 = SoftUpdateModel(policy_network_2, tau=tau)

    def _compute_loss(self, ind, states, actions, rewards, masks):
        policy_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        estimated_action_values = policy_network.value(states[0], actions[0]).squeeze()

        extrinsic_loss = self.criterion(estimated_action_values, self.bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(self.bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error

    def _loss(self, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            estimated_next_values_1, _ = self.target_network_1.action_value(next_states)
            estimated_next_values_2, _ = self.target_network_2.action_value(next_states)
            estimated_next_action_values = torch.min(estimated_next_values_1, estimated_next_values_2)
        self.bellman_action_values = trajectory_reward.clone()
        self.bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        loss, td_error = super()._loss(states, actions, rewards, masks)
        self.bellman_action_values = None
        return loss, td_error
