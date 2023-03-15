from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

class DenseCriticNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes):
        super().__init__(input_size, hidden_sizes, 1)
