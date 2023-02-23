class AbstractActor:
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
