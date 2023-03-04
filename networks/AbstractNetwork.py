import torch.nn as nn

class AbstractNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def action(self, state):
        raise NotImplementedError

    def action_value(self, state):
        raise NotImplementedError

class AbstractDenseNetwork(AbstractNetwork):
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
