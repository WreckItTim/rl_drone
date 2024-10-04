
from rewarders.rewarder import Rewarder
from component import _init_wrapper

# linear combination of rewards
class Schema(Rewarder):

    # constructor, set weights to each reward component
    @_init_wrapper
    def __init__(self, rewards_components, reward_weights):
        super().__init__()
        assert len(rewards_components) == len(reward_weights), 'not equal number of reward components and weights'
        
    # calculates rewards from agent's current state (call to when taking a step)
    def step(self, state):
        total_reward = 0
        state['reached_goal'] = False
        for idx, reward in enumerate(self._rewards):
            value, done = reward.step(state)
            weighted_value = self.reward_weights[idx] * value
            state['reward_from_' + reward._name] = weighted_value
            total_reward += weighted_value
            if done:
                break
        state['total_reward'] = total_reward
        return total_reward, done