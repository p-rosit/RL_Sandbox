class AbtractActor:
    def __init__(self):
        self.optimizer = None
        self.criterion = None

    def sample(self, state):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def parameters(self):
        raise NotImplementedError

    def step(self, experiences):
        raise NotImplementedError
