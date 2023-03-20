import torch
from core.agents.abstract_soft_actor_critic import AbstractSoftActorCritic

class SoftActorCritic(AbstractSoftActorCritic):

    def _compute_actor_loss(self, states, actions, rewards, trajectory_length):
        pass

    def _compute_critic_loss(self, ind, states, actions, rewards, trajectory_length):
        pass

    def _loss(self, states, actions, rewards, trajectory_length):
        # compute targets
        return super()._loss(states, actions, rewards, trajectory_length)
