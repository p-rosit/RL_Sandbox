import torch

class AbstractAgent:
    def __init__(self, discount=0.99, max_grad=torch.inf):
        self.training = True
        self.optimizer = None
        self.criterion = None
        self.discount = discount
        self.max_grad = max_grad

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def sample(self, state):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def parameters(self):
        raise NotImplementedError

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.max_grad)
        self.optimizer.step()

    def step(self, experiences):
        raise NotImplementedError

class AbstractOptimizerFreeAgent(AbstractAgent):
    def set_optimizer(self, _):
        raise AttributeError("Class %s does not require an optimizer." % self.__class__.__name__)

    def set_criterion(self, _):
        raise AttributeError("Class %s does not require a criterion." % self.__class__.__name__)

    def parameters(self):
        raise AttributeError("Class %s does not have any trainable parameters." % self.__class__.__name__)

    def _step(self, _):
        raise AttributeError("Class %s cannot be updated." % self.__class__.__name__)

    def step(self, _):
        raise AttributeError("Class %s cannot be updated." % self.__class__.__name__)
