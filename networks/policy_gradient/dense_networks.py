import torch
import torch.nn as nn
from torch.distributions import Categorical
from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

class DensePolicyNetwork(AbstractDenseNetwork):

    def forward(self, state):
        logits = self.network(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        return action.view(1, 1), action.item()

    def log_prob(self, state, actions):
        logits = self.network(state)

        log_probs = torch.zeros(logits.size(0), 1)
        for i, action in enumerate(actions):
            dist = Categorical(logits=logits[i])
            log_probs[i] = dist.log_prob(action)

        return log_probs


class DenseEgoMotionPolicyNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, alpha_start=100, alpha_end=0.01, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, output_size)
        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay

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

    def forward(self, state):
        logits = self.action_layer(self.initial_network(state))
        dist = Categorical(logits=logits)

        action = dist.sample()
        return action.view(1, 1), action.item()

    def log_prob(self, state, actions):
        logits = self.action_layer(self.initial_network(state))

        log_probs = torch.zeros(logits.size(0), 1)
        for i, action in enumerate(actions):
            dist = Categorical(logits=logits[i])
            log_probs[i] = dist.log_prob(action)

        return log_probs