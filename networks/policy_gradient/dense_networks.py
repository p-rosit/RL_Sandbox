import torch
import torch.nn as nn
from torch.distributions import Categorical
from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

class DensePolicyNetwork(AbstractDenseNetwork):

    def forward(self, x):
        return self.network(x)

    def action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)

        action = dist.sample()

        return action.view(-1, 1), action.item()

    def action_value(self, state):
        raise RuntimeError('Policy network does not estimate value.')

class DenseEgoMotionPolicyNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, alpha_start=100, alpha_end=0.01, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, output_size)
        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.loss_function = nn.CrossEntropyLoss()

    def pretrain_loss(self, states, actions, rewards, masks):
        intermediate_1 = self.initial_network(states[0, masks[0]])
        intermediate_2 = self.future_network(states[1, masks[0]])

        intermediate = torch.cat((intermediate_1, intermediate_2), dim=1)
        logits = self.action_classification_layer(intermediate)
        classification = self.softmax(logits)

        ego_loss = self.loss_function(classification, actions[0, masks[0]].reshape(-1))

        return ego_loss

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

    def action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)

        action = dist.sample()

        return action.view(-1, 1), action.item()

    def action_value(self, state):
        raise RuntimeError('Policy network does not estimate value.')
