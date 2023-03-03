import torch
from buffer.transitions import batch_transitions
from core.agents.abstract_agent import AbstractAgent

class AbstractQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__(max_grad=max_grad)
        self.policy_network = None
        self.target_network = None
        self.discount = discount

    def sample(self, state):
        policy_action, env_action = self.policy_network.action(state)
        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return self.policy_network.parameters()

    def _compute_loss(self, policy_network, target_network, experiences):
        raise NotImplementedError

    def _step(self, batch_experiences):
        loss = self._compute_loss(self.policy_network, self.target_network, batch_experiences)
        super()._step(loss)

    def step(self, experiences):
        experiences = batch_transitions(experiences)
        self._step(experiences)


class AbstractDoubleQLearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__(max_grad=max_grad)
        self.policy_network_1 = None
        self.policy_network_2 = None
        self.target_network_1 = None
        self.target_network_2 = None
        self.discount = discount

    def sample(self, state):
        if self.training:
            if torch.rand(1) < 0.5:
                policy_action, env_action = self.policy_network_1.action(state)
            else:
                policy_action, env_action = self.policy_network_2.action(state)
        else:
            value_1, action_1, env_action_1 = self.policy_network_1.action_value(state)
            value_2, action_2, env_action_2 = self.policy_network_2.action_value(state)
            if value_1 > value_2:
                policy_action = action_1
                env_action = env_action_1
            else:
                policy_action = action_2
                env_action = env_action_2

        return policy_action.view(-1, 1), env_action

    def parameters(self):
        return *self.policy_network_1.parameters(), *self.policy_network_2.parameters()

    def _compute_loss(self, ind, states, actions, rewards, non_final_next_states, non_final_mask):
        raise NotImplementedError

    def _step(self, experiences):
        loss_1 = self._compute_loss(0, *experiences)
        loss_2 = self._compute_loss(1, *experiences)

        super()._step(loss_1 + loss_2)

    def step(self, experiences):
        batch_experiences = batch_transitions(experiences)
        self._step(batch_experiences)

class AbstractMultiQlearningAgent(AbstractAgent):
    def __init__(self, discount=0.99, max_grad=100):
        super().__init__(max_grad=max_grad)
        self.policy_networks = None
        self.target_networks = None
        self.discount = discount

    def sample(self, state):
        if self.training:
            ind = torch.randint(len(self.policy_networks), (1,))
            policy_action, env_action = self.policy_networks[ind].action(state)
        else:
            # fix vectorization
            actions = [policy_network.get_action_value(state) for policy_network in self.policy_networks]
            _, policy_action, env_action = max(actions, key=lambda x: x[0])

        return policy_action.view(-1, 1), env_action

    def parameters(self):
        for policy_network in self.policy_networks:
            for param in policy_network.parameters():
                yield param

    def _compute_loss(self, ind, states, actions, rewards, non_final_next_states, non_final_mask):
        raise NotImplementedError

    def _step(self, experiences):
        loss = torch.tensor([0.0])
        for ind, policy_network in enumerate(self.policy_networks):
            loss += self._compute_loss(ind, *experiences)

        super()._step(loss)

    def step(self, experiences):
        batch_experiences = batch_transitions(experiences)
        self._step(batch_experiences)
