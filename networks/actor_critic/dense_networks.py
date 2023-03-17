import torch
from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

class DenseCriticNetwork(AbstractDenseNetwork):
    def __init__(self, input_size, hidden_sizes):
        super().__init__(input_size, hidden_sizes, 1)

class DenseEgoMotionCriticNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes):
        super().__init__(input_size, hidden_sizes, 1)

    def intrinsic_loss(self, states, log_probs, actions, rewards):
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
