import torch
from core.agents.abstract_policy_gradient_agent import AbstractPolicyGradientAgent

class ReinforceAgent(AbstractPolicyGradientAgent):
    def __init__(self, network, discount=0.99, max_grad=torch.inf):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = network

    def _compute_loss(self, policy_network, log_probs, rewards):
        extrinsic_loss = torch.tensor([0.0])
        for i, (episode_log_probs, episode_rewards) in enumerate(zip(log_probs, rewards)):
            size = episode_log_probs.size(0)

            discount_pows = self.discount * torch.ones(size)
            discount_pows = discount_pows.pow(torch.arange(size))

            step_discounts = torch.cat((discount_pows, torch.zeros(1)))
            step_discounts = torch.triu(step_discounts.repeat(size).view(-1, size)[:-1])

            step_discounts = torch.mm(step_discounts, episode_rewards.view(-1, 1)) * discount_pows.view(-1, 1)
            policy_function = step_discounts * episode_log_probs

            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(log_probs)
        intrinsic_loss = policy_network.intrinsic_loss(log_probs, rewards)

        return extrinsic_loss + intrinsic_loss
