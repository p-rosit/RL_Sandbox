import torch.nn as nn
from torch.distributions import Categorical
from networks.abstract_networks import AbstractDenseNetwork

class DenseQNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size)

    def forward(self, x):
        return self.network(x)

    def action(self, state):
        policy_action = self.forward(state).argmax(dim=1)
        env_action = policy_action.item()
        return policy_action, env_action

    def action_value(self, state):
        action_value, policy_action = self.forward(state).max(dim=1)
        env_action = policy_action.item()
        return action_value, policy_action, env_action

class DensePolicyNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size)
        self.prob_layer = nn.Softmax(dim=1)

    def forward(self, x):
        return self.prob_layer(self.network(x))

    def action(self, state):
        dist = Categorical(self.forward(state))

        env_action = dist.sample()
        log_prob = dist.log_prob(env_action)

        return log_prob, env_action.item()

    def action_value(self, state):
        raise RuntimeError('Policy network does not estimate value.')