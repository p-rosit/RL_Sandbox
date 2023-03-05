import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAdvantageAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _compute_loss(self, policy_network, log_probs, rewards):
        extrinsic_loss = torch.zeros(1)

        max_trajectory = max(reward.size(0) for reward in rewards)
        value_estimate = torch.zeros(max_trajectory, 1)
        trajectory_amount = torch.zeros(max_trajectory, 1)
        trajectory_rewards = []

        all_discount_pows = self.discount * torch.ones(max_trajectory)
        all_discount_pows = all_discount_pows.pow(torch.arange(max_trajectory))

        for episode_rewards in rewards:
            size = episode_rewards.size(0)

            discount_pows = all_discount_pows[:size]

            step_discounts = torch.cat((discount_pows, torch.zeros(1)))
            step_discounts = torch.triu(step_discounts.repeat(size).view(-1, size)[:-1])

            episode_trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            trajectory_rewards.append(episode_trajectory_rewards)

            value_estimate[:size] += episode_trajectory_rewards
            trajectory_amount[:size] += 1

        value_estimate /= trajectory_amount

        for episode_log_probs, episode_trajectory_rewards in zip(log_probs, trajectory_rewards):
            size = episode_log_probs.size(0)

            policy_function = (episode_trajectory_rewards - value_estimate[:size]) * episode_log_probs
            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(log_probs)
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss

class ModifiedReinforceAdvantageAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, truncate_grad_trajectory=torch.inf, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network
        self.truncate_grad_trajectory = truncate_grad_trajectory

    def _compute_loss(self, policy_network, log_probs, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)
        total_grads = min(max_trajectory, self.truncate_grad_trajectory)
        extrinsic_loss = torch.zeros(total_grads)
        samples_of_grad = torch.zeros(total_grads)

        value_estimate = torch.zeros(max_trajectory, 1)
        trajectory_amount = torch.zeros(max_trajectory, 1)
        trajectory_rewards = []

        all_discount_pows = self.discount * torch.ones(max_trajectory)
        all_discount_pows = all_discount_pows.pow(torch.arange(max_trajectory))

        for episode_rewards in rewards:
            size = episode_rewards.size(0)

            discount_pows = all_discount_pows[:size]

            step_discounts = torch.cat((discount_pows, torch.zeros(1)))
            step_discounts = torch.triu(step_discounts.repeat(size).view(-1, size)[:-1])

            episode_trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            trajectory_rewards.append(episode_trajectory_rewards)

            value_estimate[:size] += episode_trajectory_rewards
            trajectory_amount[:size] += 1

        value_estimate /= trajectory_amount

        for episode_log_probs, episode_trajectory_rewards in zip(log_probs, trajectory_rewards):
            size = episode_log_probs.size(0)

            discount_pows = all_discount_pows[:size].view(-1, 1)
            policy_function = (episode_trajectory_rewards - value_estimate[:size]) * episode_log_probs

            for k in range(min(size, self.truncate_grad_trajectory)):
                extrinsic_loss[k] -= (policy_function[k:] * discount_pows[:size-k]).sum()
                samples_of_grad[k] += 1

        extrinsic_loss = (extrinsic_loss / samples_of_grad).sum()
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss
