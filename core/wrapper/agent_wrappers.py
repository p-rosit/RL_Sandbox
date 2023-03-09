import torch
from core.agents.abstract_agent import AbstractAgent, AbstractOptimizerFreeAgent

class Discrete2Continuous(AbstractAgent):
    def __init__(self, agent, remap):
        super().__init__()
        self.agent = agent
        self.remap = remap
        self.reverse_map = {val: ind for ind, val in enumerate(self.remap)}

    def sample(self, state):
        return self.remap[self.agent.sample(state)]

    def set_optimizer(self, optimizer):
        self.agent.set_optimizer(optimizer)

    def set_criterion(self, criterion):
        self.agent.set_criterion(criterion)

    def parameters(self):
        return self.agent.parameters()

    def pretrain_loss(self, *args, **kwargs):
        return self.agent.pretrain_loss(*args, **kwargs)

    def step(self, continuous_experiences):
        discrete_experiences = []
        for state, continuous_action, reward, next_state in continuous_experiences:
            discrete_experiences.append((state, self.reverse_map[continuous_action], reward, next_state))
        self.agent.step(discrete_experiences)

    def optimizer_zero_grad(self):
        self.agent.optimizer_zero_grad()

    def optimizer_step(self):
        self.agent.optimizer_step()

class MultiAgent(AbstractOptimizerFreeAgent):
    def __init__(self, *agents: AbstractAgent, p: list = None):
        super().__init__()
        self.agents = agents
        if p is None:
            self.p = [1 / len(self.agents) for _ in self.agents]
        else:
            self.p = p
        if len(self.agents) != len(self.p):
            raise ValueError('Amount of agents and amount of probabilities need to match one to one.')

    def sample(self, state):
        ind = torch.randint(len(self.agents))
        return self.agents[ind].sample(state)

    def pretrain_loss(self, *args, **kwargs):
        return sum(agent.pretrain_loss(*args, **kwargs) for agent in self.agents)

    def step(self, *args):
        for agent in self.agents:
            agent.step(*args)

    def optimizer_zero_grad(self):
        [agent.optimizer_zero_grad() for agent in self.agents]

    def optimizer_step(self):
        [agent.optimizer_step() for agent in self.agents]

class AnnealAgent(AbstractAgent):
    def __init__(self, agent, replacement_agent, start_steps=1000, eps_start=0.9, eps_end=0.05, decay_steps=1000):
        super().__init__()
        self.agent = agent
        self.replacement_agent = replacement_agent
        self.start_steps = start_steps
        self.decay_steps = decay_steps
        self.eps_start = eps_start
        self.eps_end = eps_end

        self.curr_step = 0

    def reset_annealing(self):
        self.curr_step = 0

    def sample(self, state):
        if self.training:
            t = torch.exp(torch.tensor(-1. * (self.curr_step - self.start_steps) / self.decay_steps, dtype=torch.float64))
            eps = self.eps_end + (self.eps_start - self.eps_end) * t
            self.curr_step += 1

            if self.curr_step <= self.start_steps or torch.rand(1) < eps:
                return self.replacement_agent.sample(state)
            else:
                return self.agent.sample(state)
        else:
            return self.agent.sample(state)

    def set_optimizer(self, optimizer):
        self.agent.set_optimizer(optimizer)

    def set_criterion(self, criterion):
        self.agent.set_criterion(criterion)

    def parameters(self):
        return self.agent.parameters()

    def pretrain_loss(self, *args, **kwargs):
        return self.agent.pretrain_loss(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.agent.step(*args, **kwargs)

    def optimizer_zero_grad(self):
        self.agent.optimizer_zero_grad()

    def optimizer_step(self):
        self.agent.optimizer_step()
