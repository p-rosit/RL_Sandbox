from collections import namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def batch_transitions(experiences):
    batch_experiences = ActionTransition(*zip(*experiences))
    states = []
    actions = []
    rewards = []
    masks = []

    state, action, reward = batch_experiences
    max_trajectory = max(len(ac) for ac in action)

    for i in range(max_trajectory):
        states.append(torch.cat(
            [st[i] for st in state if len(st) > i]
        ))

    for i in range(max_trajectory-1):
        actions.append(torch.cat(
            [ac[i] for ac in action if len(ac) > i]
        ))
        rewards.append(torch.cat(
            [rw[i] for rw in reward if len(rw) > i]
        ))
        masks.append(torch.tensor(
            [len(st) > i for st in state], dtype=torch.bool
        ))

    return states, actions, rewards, masks

ActionTransition = namedtuple('ActionTransition', ('state', 'action', 'reward'))

def batch_action_transition(experiences):
    states = []
    actions = []
    rewards = []

    for episode in experiences:
        batch_experiences = ActionTransition(*zip(*episode))

        states.append(torch.cat(batch_experiences.state, dim=0))
        actions.append(torch.cat(batch_experiences.action, dim=0))
        rewards.append(torch.cat(batch_experiences.reward, dim=0))

    return states, actions, rewards
