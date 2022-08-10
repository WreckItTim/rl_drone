# discrete actions - RL returns an index specifying which action to take
from actors.actor import Actor
from random import randint
from component import _init_wrapper

class DiscreteActor(Actor):
    # constructor
    @_init_wrapper
    def __init__(self, action_names=[], name=None):
        super().__init__()

    # interpret action from RL
    def act(self, rl_output):
        self._actions[rl_output].act()

    # act on a random rl_output
    def test(self):
        random_rl_output = randint(0, len(self._actions)-1)
        print(f'randomly taking action:{self._actions[random_rl_output]._name} at index:{random_rl_output}')
        self.act(random_rl_output)