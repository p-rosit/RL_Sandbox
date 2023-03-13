import torch
from core.agents.abstract_q_learning_agent import AbstractQLearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class QLearningAgent(AbstractQLearningAgent):
    def __init__(self, policy_network, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.policy_network = policy_network
        self.target_network = SoftUpdateModel(policy_network, tau=tau)

    def _compute_loss(self, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = self.policy_network(states[0]).gather(1, actions[0].reshape(-1, 1)).squeeze()

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)

        with torch.no_grad():
            next_states = states[-1, masks[-1]]
            estimated_next_action_values, _ = self.target_network(next_states).max(dim=1)
        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[masks[-1]] += discount[-1] * estimated_next_action_values

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        intrinsic_loss = self.policy_network.intrinsic_loss(states, actions, rewards, masks)

        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()

        return extrinsic_loss + intrinsic_loss, td_error
