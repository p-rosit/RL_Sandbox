import torch.nn as nn

class AbstractNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def action(self, state):
        raise NotImplementedError

    def action_value(self, state):
        raise NotImplementedError
