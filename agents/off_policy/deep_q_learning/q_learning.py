import torch
from core.agents.abstract_q_learning_agent import AbstractQLearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class QLearningAgent(AbstractQLearningAgent):
    def __init__(self, policy_network, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.policy_network = policy_network
        self.target_network = SoftUpdateModel(policy_network, tau=tau)

    def _compute_loss(self, policy_network, target_network, states, actions, rewards, non_final_next_states, non_final_mask):
        estimated_action_values = policy_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            estimated_next_action_values, _ = self.target_network(non_final_next_states).max(dim=1)
        bellman_action_values = rewards
        bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, non_final_next_states, non_final_mask)

        return extrinsic_loss + intrinsic_loss
