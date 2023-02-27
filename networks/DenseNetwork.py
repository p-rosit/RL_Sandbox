from torch import nn
from networks.AbstractNetwork import AbstractNetwork

class DenseNetwork(AbstractNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layer_sizes = (input_size, *hidden_sizes, output_size)

        layers = []
        for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
        layers.pop()

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_action(self, state):
        return self.forward(state).argmax(dim=1)

    def get_action_value(self, state):
        return tuple(reversed(self.forward(state).max(dim=1)))
