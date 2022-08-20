# continuous actions - RL returns an index specifying which action to take
from actors.actor import Actor
from component import _init_wrapper

class ContinuousActor(Actor):
    # constructor
    @_init_wrapper
    def __init__(self, action_components=[]):
        super().__init__()
        
    # interpret action from RL
    def act(self, rl_output=None):
        for idx in range(len(rl_output)):
            self._actions[idx].act(rl_output)