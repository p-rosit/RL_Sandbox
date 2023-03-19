from core.agents.abstract_agent import AbstractAgent
from core.wrapper.network_wrappers import SoftUpdateModel

class AbstractSoftActorCritic(AbstractAgent):
    def __init__(self, actor, critic_1, critic_2, discount=0.99, tau=0.005):
        super().__init__(discount=discount)
        self.actor = actor

        self.critic_1 = critic_1
        self.target_critic_1 = SoftUpdateModel(critic_1, tau=tau)
        self.critic_2 = critic_2
        self.target_critic_2 = SoftUpdateModel(critic_2, tau=tau)
