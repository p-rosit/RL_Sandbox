import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAdvantageAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _loss(self, policy_network, states, log_probs, actions, rewards):
        extrinsic_loss = torch.zeros(1)

        max_trajectory = max(reward.size(0) for reward in rewards)
        value_estimate = torch.zeros(max_trajectory, 1)
        trajectory_amount = torch.zeros(max_trajectory, 1)
        trajectory_rewards = []

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        for episode_rewards in rewards:
            size = episode_rewards.size(0)

            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

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
        intrinsic_loss = policy_network.intrinsic_loss(states, log_probs, actions, rewards)

        return extrinsic_loss + intrinsic_loss

class ModifiedReinforceAdvantageAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, truncate_grad_trajectory=torch.inf, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network
        self.truncate_grad_trajectory = truncate_grad_trajectory

    def _loss(self, policy_network, states, log_probs, actions, rewards):
        max_trajectory = max(reward.size(0) for reward in rewards)
        total_grads = min(max_trajectory, self.truncate_grad_trajectory)
        extrinsic_loss = torch.zeros(total_grads)
        samples_of_grad = torch.zeros(total_grads)

        value_estimate = torch.zeros(max_trajectory, 1)
        trajectory_amount = torch.zeros(max_trajectory, 1)
        trajectory_rewards = []

        discount_pows = self.discount * torch.ones(max_trajectory)
        discount_pows = discount_pows.pow(torch.arange(max_trajectory))
        all_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        all_step_discounts = torch.triu(all_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        for episode_rewards in rewards:
            size = episode_rewards.size(0)

            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size]

            episode_trajectory_rewards = torch.mm(step_discounts, episode_rewards.view(-1, 1))
            trajectory_rewards.append(episode_trajectory_rewards)

            value_estimate[:size] += episode_trajectory_rewards
            trajectory_amount[:size] += 1

        value_estimate /= trajectory_amount

        for episode_log_probs, episode_trajectory_rewards in zip(log_probs, trajectory_rewards):
            size = episode_log_probs.size(0)

            size_grad_trajectory = min(size, self.truncate_grad_trajectory)
            step_discounts = all_step_discounts[:, :size]
            step_discounts = step_discounts[:size_grad_trajectory]

            policy_function = (episode_trajectory_rewards - value_estimate[:size]) * episode_log_probs

            extrinsic_loss[:size_grad_trajectory] -= (
                (torch.mm(step_discounts, policy_function))
            ).sum()
            samples_of_grad[:size_grad_trajectory] += 1

        extrinsic_loss = (extrinsic_loss / samples_of_grad).sum()
        intrinsic_loss = policy_network.intrinsic_loss(states, log_probs, actions, rewards)

        return extrinsic_loss + intrinsic_loss
