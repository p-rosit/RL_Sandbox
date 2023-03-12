class AbstractAgent:
    def __init__(self, discount=0.99):
        self.training = True
        self.optimizer = None
        self.criterion = None
        self.discount = discount

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def sample(self, state):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def _pretrain_loss(self, *args, **kwargs):
        raise NotImplementedError

    def pretrain_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _loss(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def update_target(self):
        pass
