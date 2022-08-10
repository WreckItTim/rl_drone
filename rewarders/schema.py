# penalizes colliding with objects
from rewarders.rewarder import Rewarder
from component import _init_wrapper

class Schema(Rewarder):
    # constructor
    @_init_wrapper
    def __init__(self, reward_names=[], reward_weights=[], name=None):
        super().__init__()
        
    # calculates rewards from agent's current state (call to when taking a step)
    def evalute(self, state):
        total_reward = 0
        for idx, reward in enumerate(self._rewards):
            value = reward.evaluate(state)
            state['reward_from_' + reward._name] = value
            total_reward += self.reward_weights[idx] * value
        state['total_reward'] = total_reward
        return total_reward

    def test(self):
        state = {}
        print('getting state from schema.evaluate()...')
        self.evalute(state)
        print('*** begin state ***')
        print(state)
        print('***  end state  ***')