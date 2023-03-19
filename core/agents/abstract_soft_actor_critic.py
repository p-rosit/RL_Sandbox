from torch import nn
from core.agents.abstract_agent import AbstractAgent
from core.wrapper.network_wrappers import SoftUpdateModel
from buffer.transitions import batch_episodes, batch_trajectories

class AbstractSoftActorCritic(AbstractAgent):
    def __init__(self, actor, critic_1, critic_2, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.criterion = nn.SmoothL1Loss()
        self.actor = actor

        self.critic_1 = critic_1
        self.target_critic_1 = SoftUpdateModel(critic_1, tau=tau)
        self.critic_2 = critic_2
        self.target_critic_2 = SoftUpdateModel(critic_2, tau=tau)

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        super().train()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        super().eval()

    def sample(self, state):
        policy_action, env_action = self.actor.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        for param in self.actor.parameters():
            yield param
        for param in self.critic_1.parameters():
            yield param
        for param in self.critic_2.parameters():
            yield param

    def _pretrain_loss(self, states, actions, rewards, masks):
        actor_loss = self.actor.pretrain_loss(states, actions, rewards, masks)
        critic_1_loss = self.critic_1.pretrain_loss(states, actions, rewards, masks)
        critic_2_loss = self.critic_2.pretrain_loss(states, actions, rewards, masks)
        return actor_loss + critic_1_loss + critic_2_loss

    def pretrain_loss(self, experiences):
        batch_experiences = batch_trajectories(experiences)
        return self._pretrain_loss(*batch_experiences)

    def _compute_loss(self, states, actions, rewards, trajectory_length):
        raise NotImplementedError

    def _loss(self, states, actions, rewards, trajectory_length):
        return self._compute_loss(states, actions, rewards, trajectory_length)

    def loss(self, experiences, trajectory_length=1):
        states, actions, rewards = batch_episodes(experiences)
        return self._loss(states, actions, rewards, trajectory_length)

    def update_target(self):
        self.target_critic_1.update(self.critic_1)
        self.target_critic_2.update(self.critic_2)
