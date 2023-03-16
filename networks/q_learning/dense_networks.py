import torch
import torch.nn as nn
from networks.abstract_networks import AbstractDenseNetwork, AbstractDenseEgoMotionNetwork

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

class DenseEgoMotionQNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, alpha_start=1, alpha_end=0, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, output_size)
        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.loss_function = nn.CrossEntropyLoss()

    def intrinsic_loss(self, states, actions, rewards, masks):
        t = torch.exp(torch.tensor(-1. * self.curr_step / self.alpha_decay, dtype=torch.float64))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * t
        self.curr_step += 1

        return alpha * self.pretrain_loss(states, actions, rewards, masks)

    def action(self, state):
        policy_action = self.forward(state).argmax(dim=1)
        env_action = policy_action.item()
        return policy_action, env_action

    def action_value(self, state):
        action_value, policy_action = self.forward(state).max(dim=1)
        env_action = policy_action.item()
        return action_value, policy_action, env_action
