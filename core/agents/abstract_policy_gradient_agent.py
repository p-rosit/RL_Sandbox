from core.agents.abstract_agent import AbstractAgent

class AbstractPolicyGradientAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__(discount=discount, max_grad=max_grad)

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, experiences):
        loss = 0
        super()._step(loss)

    def step(self, experiences):
        self._step(experiences)
