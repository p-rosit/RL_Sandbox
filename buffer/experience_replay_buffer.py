import numpy as np
from buffer.transitions import Experience
from buffer.abstract_buffer import AbstractBuffer

rng = np.random.default_rng()

class PrioritizedExperienceReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=100, default_weight=10.0):
        super().__init__(max_size=max_size)
        self.buffer = []
        self.weights = []
        self.default_prob = default_weight

        self.episode_inds = []
        self.inds = []

        self.episode = []

    def __len__(self):
        return sum(episode.state.shape[0] for episode in self.buffer)

    def _normalize_weights(self):
        episode_sum = [episode.sum() for episode in self.weights]
        total_sum = sum(episode_sum)

        episode_weights = [weight / total_sum for weight in episode_sum]
        normalized_weights = [episode / ep_sum for ep_sum, episode in zip(episode_sum, self.weights)]
        return episode_weights, normalized_weights

    def update_weights(self, sample_errors):
        for i, (episode_ind, ind) in enumerate(zip(self.episode_inds, self.inds)):
            self.weights[episode_ind][ind] = sample_errors[i]

    def append(self, state, action, reward, episode_terminated=False):
        experience = Experience(state.numpy(), action.numpy(), reward.numpy())
        self.episode.append(experience)

        if episode_terminated:
            self.episode.append(Experience(
                np.zeros_like(experience.state),
                np.zeros_like(experience.action),
                np.zeros_like(experience.reward)
            ))
            states, actions, rewards = zip(*self.episode)
            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            episode = Experience(states, actions, rewards)

            self.buffer.append(episode)
            self.weights.append(np.full(len(self.episode) - 1, self.default_prob))

            self.episode = []
            while len(self) > self.max_size and len(self.buffer) > 1:
                self.buffer.pop(0)
                self.weights.pop(0)

    def sample(self, batch_size=1, trajectory_length=1):
        episode_weights, normalized_weights = self._normalize_weights()

        self.episode_inds = rng.choice(len(self.buffer), batch_size, p=episode_weights)
        self.sample_amount = []
        for i in range(len(self.buffer)):
            self.sample_amount.append((self.episode_inds == i).sum())
        self.inds = []

        trajectories = []
        for episode_ind, episode_sample_amount in enumerate(self.sample_amount):
            episode = self.buffer[episode_ind]
            ind = rng.choice(episode.state.shape[0] - 1, episode_sample_amount, p=normalized_weights[episode_ind])
            self.inds.append(ind)

            print(ind)
            print(ind + trajectory_length)

            indices = np.arange(ind, ind + trajectory_length)

            print(indices)

            # print(episode)
            # print(ind)
            error(':)')

            episode_states, episode_actions, episode_rewards = episode

            if episode_states.shape[0] > ind + trajectory_length:
                states = episode_states[ind:(ind+trajectory_length+1)]
            else:
                states = episode_states[ind:(ind+trajectory_length)]
            actions = episode_actions[ind:min(episode_actions.shape[0], ind+trajectory_length)]
            rewards = episode_rewards[ind:min(episode_rewards.shape[0], ind+trajectory_length)]

            trajectories.append(Experience(states, actions, rewards))

        return trajectories

    def all_episodes(self):
        return self.buffer

    def clear(self):
        self.buffer = []
        self.weights = []
        self.episode = []

class ReplayBuffer(AbstractBuffer):
    def __init__(self, max_size=100, default_weight=10.0):
        super().__init__(max_size=max_size)
        self.ind = 0
        self.buffer = [None for _ in range(max_size)]
        self.weights = np.zeros(max_size)
        self.default_prob = default_weight

        self.half_experience = None
        self.inds = []

    def __len__(self):
        return sum(1 for _ in self.buffer if _ is not None)

    def _normalize_weights(self):
        return self.weights / self.weights.sum()

    def update_weights(self, sample_errors):
        self.weights[self.inds] = sample_errors

    def _make_experience(self, state, action, reward, next_state):
        if next_state is not None:
            states = np.concatenate((state, next_state), axis=0)
        else:
            states = state

        return Experience(states, action, reward)

    def append(self, state, action, reward, episode_terminated=False):
        state = state.numpy()
        action = action.numpy()
        reward = reward.numpy()

        if self.half_experience is not None:
            self.weights[self.ind] = self.default_prob
            self.buffer[self.ind] = self._make_experience(*self.half_experience, state)
            self.ind = (self.ind + 1) % self.max_size

        if episode_terminated:
            self.weights[self.ind] = self.default_prob
            self.buffer[self.ind] = self._make_experience(state, action, reward, None)
            self.ind = (self.ind + 1) % self.max_size

            self.half_experience = None
        else:
            self.half_experience = (state, action, reward)

    def sample(self, batch_size=1, trajectory_length=1):
        normalized_weights = self._normalize_weights()

        self.inds = rng.choice(self.max_size, batch_size, p=normalized_weights)
        return [self.buffer[ind] for ind in self.inds]

    def all_episodes(self):
        raise RuntimeError('Buffer cannot be used for online training.')

    def clear(self):
        self.buffer = []
        self.weights = []

if __name__ == '__main__':
    b = PrioritizedExperienceReplayBuffer(max_size=2)

    b.append([1, 2], 2, 3)
    b.append([2, 3], 3, 4)
    b.append([3, 4], 4, 5)

    print(b.buffer)
    print(b.sample(2))
