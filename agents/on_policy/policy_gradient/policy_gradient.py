import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAgent(AbstractPolicyGradientAgent):

    def _compute_loss(self, states, actions, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        extrinsic_loss = torch.zeros(1)
        for episode_states, episode_actions, episode_rewards in zip(states, actions, rewards):
            size = episode_states.size(0)

            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

            trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            policy_function = trajectory_rewards * self.policy_network.log_prob(episode_states, episode_actions)

            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(states)
        intrinsic_loss = self.policy_network.intrinsic_loss(states, actions, rewards)

        return extrinsic_loss + intrinsic_loss

class ModifiedReinforceAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, truncate_grad_trajectory=torch.inf, discount=0.99):
        super().__init__(network, discount=discount)
        self.truncate_grad_trajectory = truncate_grad_trajectory

    def _compute_loss(self, states, actions, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)
        total_grads = min(max(episode_states.size(0) for episode_states in states), self.truncate_grad_trajectory)

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        extrinsic_loss = torch.zeros(total_grads)
        samples_of_grad = torch.zeros(total_grads)

        for episode_states, episode_actions, episode_rewards in zip(states, actions, rewards):
            size = episode_states.size(0)

            size_grad_trajectory = min(size, self.truncate_grad_trajectory)
            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

            trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            policy_function = trajectory_rewards * self.policy_network.log_prob(episode_states, episode_actions)

            step_discounts = step_discounts[:size_grad_trajectory]

            extrinsic_loss[:size_grad_trajectory] -= (
                (torch.mm(step_discounts, policy_function))
            ).sum()
            samples_of_grad[:size_grad_trajectory] += 1

        extrinsic_loss = (extrinsic_loss / samples_of_grad).sum()
        intrinsic_loss = self.policy_network.intrinsic_loss(states, actions, rewards)

        return extrinsic_loss + intrinsic_loss
