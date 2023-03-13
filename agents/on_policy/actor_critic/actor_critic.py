import torch
from core.agents.abstract_actor_critic_agent import AbstractActorCriticAgent

class ActorCriticAgent(AbstractActorCriticAgent):

    def _compute_loss(self, states, log_probs, actions, rewards):
        raise NotImplementedError
