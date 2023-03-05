import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _compute_loss(self, policy_network, log_probs, rewards):
        extrinsic_loss = torch.zeros(1)

        for episode_log_probs, episode_rewards in zip(log_probs, rewards):
            size = episode_log_probs.size(0)

            discount_pows = self.discount * torch.ones(size)
            discount_pows = discount_pows.pow(torch.arange(size))

            step_discounts = torch.cat((discount_pows, torch.zeros(1)))
            step_discounts = torch.triu(step_discounts.repeat(size).view(-1, size)[:-1])

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
        total_grads = min(max(log_prob.size(0) for log_prob in log_probs), self.truncate_grad_trajectory)
        extrinsic_loss = torch.zeros(total_grads)
        samples_of_grad = torch.zeros(total_grads)

        for episode_log_probs, episode_rewards in zip(log_probs, rewards):
            size = episode_log_probs.size(0)

            discount_pows = self.discount * torch.ones(size)
            discount_pows = discount_pows.pow(torch.arange(size))

            step_discounts = torch.cat((discount_pows, torch.zeros(1)))
            step_discounts = torch.triu(step_discounts.repeat(size).view(-1, size)[:-1])

            trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            policy_function = trajectory_rewards * episode_log_probs

            for k in range(min(size, self.truncate_grad_trajectory)):
                extrinsic_loss[k] -= (policy_function[k:] * discount_pows.view(-1, 1)[:size-k]).sum()
                samples_of_grad[k] += 1

        extrinsic_loss = (extrinsic_loss / samples_of_grad).sum()
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss
