from copy import deepcopy

class SoftUpdateModel:
    def __init__(self, network, tau=0.005):
        self.actor = deepcopy(network)
        self.tau = tau

    def __call__(self, x):
        return self.actor(x)

    def update(self, actor):
        policy_state_dict = actor.state_dict()
        target_state_dict = self.actor.state_dict()
        try:
            for key in target_state_dict:
                target_state_dict[key] = self.tau * policy_state_dict[key] + (1 - self.tau) * target_state_dict[key]
            self.actor.load_state_dict(target_state_dict)
        except KeyError:
            raise KeyError('Saved model and new model do not have the same parameters.')
