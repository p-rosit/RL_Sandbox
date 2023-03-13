import torch
from core.agents.abstract_actor_critic_agent import AbstractActorCriticAgent

class ActorCriticAgent(AbstractActorCriticAgent):

    def _compute_loss(self, states, log_probs, advantages, actions, rewards, trajectory_length):
        extrinsic_loss = torch.zeros(1)

        max_trajectory = max(reward.size(0) for reward in rewards)

        discount_pows = self.discount * torch.ones(trajectory_length)
        discount_pows = discount_pows.pow(torch.arange(trajectory_length))
        discount_pows = torch.cat((discount_pows, torch.zeros(max_trajectory - trajectory_length)))
        truncated_step_discounts = torch.cat((discount_pows, torch.zeros(1)))
        truncated_step_discounts = torch.triu(truncated_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        print(truncated_step_discounts)
        print(discount_pows)
        error(':)')

        for _ in range(5):
            pass

        for episode_log_probs, episode_advantages in zip(log_probs, advantages):
            policy_function = episode_advantages * episode_log_probs
            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(log_probs)
        intrinsic_loss = self.actor_critic.intrinsic_loss(states, log_probs, advantages, actions, rewards)

        return extrinsic_loss + intrinsic_loss
