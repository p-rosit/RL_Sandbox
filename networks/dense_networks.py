import torch
import torch.nn as nn
from torch.distributions import Categorical
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
    def __init__(self, input_size, hidden_sizes, output_size, alpha_start=100, alpha_end=0.01, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, output_size)
        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.loss_function = nn.CrossEntropyLoss()

    def intrinsic_loss(self, states, actions, rewards, non_final_next_states, non_final_mask):
        intermediate_1 = self.initial_network(states[non_final_mask])
        intermediate_2 = self.initial_network(non_final_next_states)

        intermediate = torch.cat((intermediate_1, intermediate_2), dim=1)
        logits = self.action_classification_layer(intermediate)
        classification = self.softmax(logits)

        intrinsic_loss = self.loss_function(classification, actions[non_final_mask].reshape(-1))

        t = torch.exp(torch.tensor(-1. * self.curr_step / self.alpha_decay, dtype=torch.float64))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * t
        self.curr_step += 1
        return alpha * intrinsic_loss

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

class DenseEgoMotionPolicyNetwork(AbstractDenseEgoMotionNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, alpha_start=100, alpha_end=0.01, alpha_decay=1000):
        super().__init__(input_size, hidden_sizes, output_size)
        self.prob_layer = nn.Softmax(dim=1)

        self.curr_step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.prob_layer(super().forward(x))

    def intrinsic_loss(self, states, log_probs, actions, rewards):
        intrinsic_loss = torch.zeros(1)

        for episode_states, episode_actions in zip(states, actions):
            intermediate_1 = self.initial_network(episode_states[:-1])
            intermediate_2 = self.initial_network(episode_states[1:])

            intermediate = torch.cat((intermediate_1, intermediate_2), dim=1)
            logits = self.action_classification_layer(intermediate)
            classification = self.softmax(logits)

            intrinsic_loss += self.loss_function(classification, episode_actions[:-1].reshape(-1))

        t = torch.exp(torch.tensor(-1. * self.curr_step / self.alpha_decay, dtype=torch.float64))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * t
        self.curr_step += 1

        return alpha * intrinsic_loss

    def action(self, state):
        dist = Categorical(self.forward(state))

        env_action = dist.sample()
        log_prob = dist.log_prob(env_action)

        return log_prob, env_action.item()

    def action_value(self, state):
        raise RuntimeError('Policy network does not estimate value.')
