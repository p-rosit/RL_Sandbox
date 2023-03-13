import torch
from torch.distributions import Categorical
from buffer.transitions import batch_trajectories, batch_episodes
from core.agents.abstract_agent import AbstractAgent

class AbstractPolicyGradientAgent(AbstractAgent):
    def __init__(self, policy_network, discount=0.99):
        super().__init__(discount=discount)
        self.policy_network = policy_network

    def train(self):
        self.policy_network.train()
        super().train()
    
    def eval(self):
        self.policy_network.eval()
        super().eval()

    def sample(self, state):
        policy_action, env_action = self.policy_network.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return self.policy_network.parameters()

    def _pretrain_loss(self, states, actions, rewards, masks):
        return self.policy_network.pretrain_loss(states, actions, rewards, masks)

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _compute_loss(self, policy_network, states, log_probs, actions, reward):
        raise NotImplementedError

    def _loss(self, states, log_probs, actions, rewards):
        return self._compute_loss(self.policy_network, states, log_probs, actions, rewards)

    def loss(self, experiences):
        states, actions, rewards = batch_episodes(experiences)

        log_probs = []
        for episode_states, episode_actions in zip(states, actions):
            logits = self.policy_network(episode_states)
            episode_log_probs = torch.zeros(logits.size(0), 1)

            for i, action in enumerate(episode_actions):
                dist = Categorical(logits=logits[i])
                episode_log_probs[i] = dist.log_prob(action)

            log_probs.append(episode_log_probs)

        return self._loss(states, log_probs, actions, rewards)
