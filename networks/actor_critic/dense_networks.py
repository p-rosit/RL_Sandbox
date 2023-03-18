import torch
from torch import nn
from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

class DenseCriticNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes):
        super().__init__(input_size, hidden_sizes, 1)

class DenseEgoMotionCriticNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes, alpha_start=1, alpha_end=0, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, 1, action_size=2)
        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.loss_function = nn.CrossEntropyLoss()

    def intrinsic_loss(self, states, actions, rewards):
        t = torch.exp(torch.tensor(-1. * self.curr_step / self.alpha_decay, dtype=torch.float64))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * t
        self.curr_step += 1

        ego_loss = torch.zeros(1)
        for state, action in zip(states, actions):
            intermediate_1 = self.initial_network(state[:-1])
            intermediate_2 = self.future_network(state[1:])

            intermediate = torch.cat((intermediate_1, intermediate_2), dim=1)
            logits = self.action_classification_layer(intermediate)
            classification = self.softmax(logits)

            ego_loss += self.loss_function(classification, action[:-1].reshape(-1))

        return alpha * ego_loss
