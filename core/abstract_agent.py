class AbstractAgent:
    def __init__(self):
        self.training = True
        self.optimizer = None
        self.criterion = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def sample(self, state):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def parameters(self):
        raise NotImplementedError

    def step(self, experiences):
        raise NotImplementedError

class AbstractOptimizerFreeAgent(AbstractAgent):
    def set_optimizer(self, optimizer):
        raise AttributeError("Class %s does not require an optimizer." % self.__class__.__name__)

    def set_criterion(self, criterion):
        raise AttributeError("Class %s does not require a criterion." % self.__class__.__name__)

    def parameters(self):
        raise AttributeError("Class %s does not have any parameters." % self.__class__.__name__)