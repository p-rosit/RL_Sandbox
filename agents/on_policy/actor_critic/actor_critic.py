import torch
from torch.distributions import Categorical
from core.agents.abstract_actor_critic_agent import AbstractActorCriticAgent

class ActorCriticAgent(AbstractActorCriticAgent):

    def _compute_loss(self, states, actions, rewards, trajectory_length):
        extrinsic_loss = torch.zeros(1)

        max_trajectory = max(reward.size(0) for reward in rewards)

        discount_pows = self.discount * torch.ones(trajectory_length + 1)
        discount_pows = discount_pows.pow(torch.arange(trajectory_length + 1))
        truncated_pows = torch.cat((discount_pows[:-1], torch.zeros(max_trajectory - trajectory_length)))
        truncated_step_discounts = torch.cat((truncated_pows, torch.zeros(1)))
        truncated_step_discounts = torch.triu(truncated_step_discounts.repeat(max_trajectory).view(-1, max_trajectory)[:-1])

        log_probs = []
        advantages = []
        for episode_states, episode_rewards, episode_actions in zip(states, rewards, actions):
            logits = self.actor(episode_states)
            values = self.target_critic.value(episode_states, None)
            print(values)
            error(':)')
            episode_log_probs = torch.zeros(logits.size(0), 1)

            for i, action in enumerate(episode_actions):
                dist = Categorical(logits=logits[i])
                episode_log_probs[i] = dist.log_prob(action)

            episode_advantages = episode_rewards + values
            episode_advantages[:-1] -= values[1:]

            log_probs.append(episode_log_probs)
            advantages.append(episode_advantages.detach())

        for episode_states, episode_rewards in zip(states, rewards):
            _, estimated_state_values = self.actor_critic(episode_states)

            size = episode_rewards.size(0)
            truncated_episode_discount = truncated_step_discounts[:, :size]
            truncated_episode_discount = truncated_episode_discount[:size]

            bellman_state_values = torch.mm(truncated_episode_discount, episode_rewards.view(-1, 1))
            bellman_state_values[:-trajectory_length-1] += discount_pows[-1] * estimated_state_values[trajectory_length+1:]
            bellman_state_values = bellman_state_values.detach()

            extrinsic_loss += self.criterion(estimated_state_values, bellman_state_values)

        for episode_log_probs, episode_advantages in zip(log_probs, advantages):
            policy_function = episode_advantages * episode_log_probs
            extrinsic_loss -= policy_function.sum()

        extrinsic_loss /= len(log_probs)
        intrinsic_loss = self.actor_critic.intrinsic_loss(states, log_probs, advantages, actions, rewards)

        return extrinsic_loss + intrinsic_loss
