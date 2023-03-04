from collections import namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def batch_transitions(experiences):
    batch_experiences = Transition(*zip(*experiences))

    states = torch.cat(batch_experiences.state, dim=0)
    actions = torch.cat(batch_experiences.action, dim=0)
    rewards = torch.cat(batch_experiences.reward, dim=0)

    non_final_next_states = torch.cat(
        [next_state for next_state in batch_experiences.next_state if next_state is not None])
    non_final_mask = torch.tensor([next_state is not None for next_state in batch_experiences.next_state],
                                  dtype=torch.bool)

    return states, actions, rewards, non_final_next_states, non_final_mask

ActionTransition = namedtuple('ActionTransition', ('action', 'reward'))

def batch_action_transition(experiences):
    batch_experiences = ActionTransition(*zip(*experiences))

    actions = torch.cat(batch_experiences.action, dim=0)
    rewards = torch.cat(batch_experiences.reward, dim=0)

    return actions, rewards
