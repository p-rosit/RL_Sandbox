from networks.AbstractNetwork import AbstractDenseNetwork

class DensePolicyNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size)

    def forward(self, x):
        return self.network(x)

    def action(self, state):
        policy_action = self.forward(state).argmax(dim=1)
        env_action = policy_action.item()
        return policy_action, env_action

    def action_value(self, state):
        raise RuntimeError('Policy network does not estimate value.')
