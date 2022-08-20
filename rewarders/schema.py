# penalizes colliding with objects
from rewarders.rewarder import Rewarder
from component import _init_wrapper

class Schema(Rewarder):
    # constructor
    @_init_wrapper
    def __init__(self, reward_components, reward_weights):
        super().__init__()
        assert len(reward_components) == len(reward_weights), 'not equal number of reward components and weights'
        self._min_reward = -1. * sum(reward_weights)
        self._max_reward = sum(reward_weights)
        self._diff = self._max_reward - self._min_reward
        
    # calculates rewards from agent's current state (call to when taking a step)
    def evaluate(self, state):
        total_reward = 0
        for idx, reward in enumerate(self._rewards):
            value = reward.evaluate(state)
            state['reward_from_' + reward._name] = value
            total_reward += self.reward_weights[idx] * value
        # normalize total reward between [-1, 1]
        total_reward = 2 * (total_reward - self._min_reward) / self._diff - 1
        state['total_reward'] = total_reward