import torch
from core.agents.abstract_q_learning_agent import AbstractQLearningAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class QLearningAgent(AbstractQLearningAgent):
    def __init__(self, policy_network, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.policy_network = policy_network
        self.target_network = SoftUpdateModel(policy_network, tau=tau)

    def _compute_loss(self, policy_network, target_network, states, actions, rewards, masks):
        discount = torch.pow(self.discount, torch.arange(len(masks) + 1)).reshape(-1, 1)
        estimated_action_values = policy_network(states[0]).gather(1, actions[0].reshape(-1, 1)).squeeze()

        # print(discount)
        # print(states)
        # print(actions)
        # print(rewards)
        # print(masks)

        trajectory_reward = (discount[:-1] * rewards).sum(dim=0)
        # print(trajectory_reward)

        final_state_indices = masks[-1]
        # print(final_state_indices)
        # error(':)')
        # print(next_state_indices)

        with torch.no_grad():
            # next_states = torch.cat([state[ind] for ind, state in zip(next_state_indices, states)], dim=0)
            # next_states = states[next_state_indices, torch.arange(masks.size(1))]
            next_states = states[-1, final_state_indices]
            # print(next_states)
            estimated_next_action_values, _ = self.target_network(next_states).max(dim=1)
        # print(discount)
        # print(discount[next_state_indices])
        # print(next_states)
        # print(estimated_next_action_values)
        bellman_action_values = trajectory_reward.clone()
        bellman_action_values[final_state_indices] += discount[-1] * estimated_next_action_values

        # print(next_states)
        # print(estimated_action_values)
        # print(bellman_action_values)

        extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        td_error = torch.abs(bellman_action_values - estimated_action_values).detach()
        # print(extrinsic_loss)
        #
        # error(':)')
        # print(td_error)
        # error(':)')
        return extrinsic_loss, td_error

        # error(':)')
        #
        # with torch.no_grad():
        #     estimated_next_action_values, _ = self.target_network(non_final_next_states).max(dim=1)
        # bellman_action_values = rewards
        # bellman_action_values[non_final_mask] += self.discount * estimated_next_action_values
        #
        # extrinsic_loss = self.criterion(estimated_action_values, bellman_action_values)
        # intrinsic_loss = policy_network.intrinsic_loss(states, actions, rewards, non_final_next_states, non_final_mask)
        #
        # return extrinsic_loss + intrinsic_loss
