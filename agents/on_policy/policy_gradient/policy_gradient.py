import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _compute_loss(self, policy_network, log_probs, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        extrinsic_loss = torch.zeros(1)
        for episode_log_probs, episode_rewards in zip(log_probs, rewards):
            size = episode_log_probs.size(0)

            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

            trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            policy_function = trajectory_rewards * episode_log_probs

            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(log_probs)
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss

class ModifiedReinforceAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, truncate_grad_trajectory=torch.inf, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network
        self.truncate_grad_trajectory = truncate_grad_trajectory

    def _compute_loss(self, policy_network, log_probs, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)
        total_grads = min(max(log_prob.size(0) for log_prob in log_probs), self.truncate_grad_trajectory)

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        extrinsic_loss = torch.zeros(total_grads)
        samples_of_grad = torch.zeros(total_grads)

        for episode_log_probs, episode_rewards in zip(log_probs, rewards):
            size = episode_log_probs.size(0)

            size_grad_trajectory = min(size, self.truncate_grad_trajectory)
            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

            trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            policy_function = trajectory_rewards * episode_log_probs

            step_discounts = step_discounts[:size_grad_trajectory]

            extrinsic_loss[:size_grad_trajectory] -= (
                (torch.mm(step_discounts, policy_function))
            ).sum()
            samples_of_grad[:size_grad_trajectory] += 1

        extrinsic_loss = (extrinsic_loss / samples_of_grad).sum()
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss
