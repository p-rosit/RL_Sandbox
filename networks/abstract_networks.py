import torch
import torch.nn as nn

class AbstractNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def pretrain_loss(self, *args, **kwargs):
        return torch.tensor([0.0], requires_grad=True)

    def intrinsic_loss(self, *arg, **kwargs):
        return torch.tensor([0.0], requires_grad=True)

    def action(self, state):
        raise NotImplementedError

    def value(self, state, action):
        raise NotImplementedError

    def action_value(self, state):
        raise NotImplementedError

class AbstractDenseNetwork(AbstractNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.network = self._make_network(input_size, hidden_sizes, output_size)

    def _make_network(self, input_size, hidden_sizes, output_size):
        layer_sizes = (input_size, *hidden_sizes, output_size)

        layers = []
        for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
        layers.pop()

        return nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

class AbstractDenseEgoMotionNetwork(AbstractNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, action_size=None):
        super().__init__()
        layer_sizes = (input_size, *hidden_sizes)

        initial_layers = []
        future_layers = []
        for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            initial_layers.append(nn.Linear(size_in, size_out))
            initial_layers.append(nn.ReLU())
            future_layers.append(nn.Linear(size_in, size_out))
            future_layers.append(nn.ReLU())

        self.initial_network = nn.Sequential(*initial_layers)
        self.future_network = nn.Sequential(*future_layers)
        self.action_layer = nn.Linear(hidden_sizes[-1], output_size)

        if action_size is None:
            action_size = output_size
        self.action_classification_layer = nn.Linear(2 * hidden_sizes[-1], action_size)
        self.softmax = nn.Softmax(dim=1)

        self.loss_function = nn.CrossEntropyLoss()

    def pretrain_loss(self, states, actions, rewards, masks):
        intermediate_1 = self.initial_network(states[0, masks[0]])
        intermediate_2 = self.future_network(states[1, masks[0]])

        intermediate = torch.cat((intermediate_1, intermediate_2), dim=1)
        logits = self.action_classification_layer(intermediate)
        classification = self.softmax(logits)

        ego_loss = self.loss_function(classification, actions[0, masks[0]])

        return ego_loss

    def intrinsic_loss(self, *arg, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.action_layer(self.initial_network(*args, **kwargs))
