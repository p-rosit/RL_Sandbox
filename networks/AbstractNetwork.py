import torch.nn as nn

class AbstractNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        raise NotImplementedError

    def get_action_value(self, state):
        raise NotImplementedError
