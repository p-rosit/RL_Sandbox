import torch
from torch.distributions import Categorical
from buffer.transitions import batch_trajectories, batch_episodes
from core.agents.abstract_agent import AbstractAgent

class AbstractActorCriticAgent(AbstractAgent):
    def __init__(self, network, discount=0.99):
        super().__init__(discount=discount)
        self.actor_critic = network

    def train(self):
        self.actor_critic.train()
        super().train()

    def eval(self):
        self.actor_critic.eval()
        super().eval()

    def sample(self, state):
        policy_action, env_action = self.actor_critic.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return self.actor_critic.parameters()

    def _pretrain_loss(self, states, actions, rewards, masks):
        return self.actor_critic.pretrain_loss(states, actions, rewards, masks)

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _compute_loss(self, states, log_probs, actions, rewards):
        raise NotImplementedError

    def _loss(self, states, log_probs, actions, rewards):
        return self._compute_loss(states, log_probs, actions, rewards)

    def loss(self, experiences):
        states, actions, rewards = batch_episodes(experiences)

        log_probs = []
        for episode_states, episode_actions in zip(states, actions):
            logits, _ = self.actor_critic(episode_states)
            episode_log_probs = torch.zeros(logits.size(0), 1)

            for i, action in enumerate(episode_actions):
                dist = Categorical(logits=logits[i])
                episode_log_probs[i] = dist.log_prob(action)

            log_probs.append(episode_log_probs)

        return self._loss(states, log_probs, actions, rewards)
