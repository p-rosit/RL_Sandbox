import torch
from torch.distributions import Categorical
from buffer.transitions import batch_action_transition
from core.agents.abstract_agent import AbstractAgent

class AbstractPolicyGradientAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)
        self.policy_network = None

    def sample(self, state):
        policy_action, env_action = self.policy_network.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return self.policy_network.parameters()

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, batch_experiences):
        loss = self._compute_loss(self.policy_network, *batch_experiences)
        super()._step(loss)

    def step(self, experiences):
        batch_experiences = batch_action_transition(experiences)
        states, actions, rewards = batch_experiences

        log_probs = []
        for episode_states, episode_actions in zip(states, actions):
            logits = self.policy_network(episode_states)
            episode_log_probs = torch.zeros(logits.size(0), 1)

            for i, action in enumerate(episode_actions):
                dist = Categorical(logits=logits[i])
                episode_log_probs[i] = dist.log_prob(action)

            log_probs.append(episode_log_probs)

        self._step((states, log_probs, actions, rewards))
