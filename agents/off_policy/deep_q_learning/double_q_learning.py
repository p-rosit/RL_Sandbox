import torch
from core.agents.abstract_q_learning_agent import AbstractQLearningAgent, AbstractDoubleQLearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class DoubleQLearningAgent(AbstractDoubleQLearningAgent):
    def __init__(self, policy_network_1, policy_network_2, discount=0.99, tau=0.005, max_grad=100, policy_train=True):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_train = policy_train
        self.policy_network_1 = policy_network_1
        self.target_network_1 = SoftUpdateModel(policy_network_1, tau=tau)

        self.policy_network_2 = policy_network_2
        self.target_network_2 = SoftUpdateModel(policy_network_2, tau=tau)

    def _compute_loss(self, ind, states, actions, rewards, non_final_next_states, non_final_mask):
        policy_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        target_network = self.target_network_2 if ind == 0 else self.target_network_1
        if self.policy_train:
            action_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        else:
            action_network = self.target_network_1 if ind == 0 else self.target_network_2

        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            estimated_next_actions = action_network(non_final_next_states).argmax(dim=1).view(-1, 1)
            estimated_next_values = target_network(non_final_next_states)

            estimated_next_action_values = estimated_next_values.gather(1, estimated_next_actions).squeeze()

        bellman_action_values = rewards.clone()
        bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, non_final_next_states, non_final_mask)

        return extrinsic_loss + intrinsic_loss

    def _step(self, experiences):
        super()._step(experiences)
        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)

class ModifiedDoubleQLearningAgent(AbstractQLearningAgent):
    def __init__(self, policy_network, discount=0.99, tau=0.005, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = policy_network
        self.target_network = SoftUpdateModel(policy_network, tau=tau)

    def _compute_loss(self, policy_network, target_network, states, actions, rewards, non_final_next_states, non_final_mask):
        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        estimated_next_action_values = torch.zeros_like(rewards)
        with torch.no_grad():
            estimated_next_actions = policy_network(non_final_next_states).argmax(dim=1).view(-1, 1)
            estimated_next_values = target_network(non_final_next_states)

            estimated_next_action_values[non_final_mask] = estimated_next_values.gather(1, estimated_next_actions).squeeze()

        bellman_action_values = rewards + self.discount * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, non_final_next_states, non_final_mask)

        return extrinsic_loss + intrinsic_loss

    def _step(self, experiences):
        super()._step(experiences)
        self.target_network.update(self.policy_network)

class ClippedDoubleQLearning(AbstractDoubleQLearningAgent):
    def __init__(self, policy_network_1, policy_network_2, discount=0.99, tau=0.005, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)
        self.bellman_action_values = None

        self.policy_network_1 = policy_network_1
        self.target_network_1 = SoftUpdateModel(policy_network_1, tau=tau)

        self.policy_network_2 = policy_network_2
        self.target_network_2 = SoftUpdateModel(policy_network_2, tau=tau)

    def _compute_loss(self, ind, states, actions, rewards, non_final_next_states, non_final_mask):
        policy_network = self.policy_network_1 if ind == 0 else self.policy_network_2
        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        extrinsic_loss = self.criterion(estimated_action_values, self.bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, non_final_next_states, non_final_mask)

        return extrinsic_loss + intrinsic_loss

    def _step(self, experiences):
        _, actions, rewards, non_final_next_states, non_final_mask = experiences

        with torch.no_grad():
            estimated_next_values_1, _ = self.target_network_1(non_final_next_states).max(dim=1)
            estimated_next_values_2, _ = self.target_network_2(non_final_next_states).max(dim=1)
            estimated_next_action_values = torch.min(estimated_next_values_1, estimated_next_values_2)
        self.bellman_action_values = rewards.clone()
        self.bellman_action_values[non_final_mask] += estimated_next_action_values

        super()._step(experiences)
        self.target_network_1.update(self.policy_network_1)
        self.target_network_2.update(self.policy_network_2)
        self.bellman_action_values = None
