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
        loss = self._compute_loss(self.policy_network, batch_experiences)
        super()._step(loss)

    def step(self, experiences):
        batch_experiences = batch_action_transition(experiences)
        self._step(experiences)
