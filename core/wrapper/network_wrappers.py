from copy import deepcopy
from networks.abstract_networks import AbstractNetwork

class SoftUpdateModel(AbstractNetwork):
    def __init__(self, network, tau=0.005):
        super().__init__()
        self.model = deepcopy(network)
        self.tau = tau

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update(self, actor):
        policy_state_dict = actor.state_dict()
        target_state_dict = self.model.state_dict()
        try:
            for key in target_state_dict:
                target_state_dict[key] = self.tau * policy_state_dict[key] + (1 - self.tau) * target_state_dict[key]
            self.model.load_state_dict(target_state_dict)
        except KeyError:
            raise KeyError('Saved model and new model do not have the same parameters.')
