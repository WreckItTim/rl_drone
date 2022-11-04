
from rewarders.rewarder import Rewarder
from component import _init_wrapper

# linear combination of rewards
class DecaySchema(Rewarder):

    # constructor, set weights to each reward component
    @_init_wrapper
    def __init__(self, rewards_components, reward_weights, decay_factor, max_steps):
        super().__init__()
        assert len(rewards_components) == len(reward_weights), 'not equal number of reward components and weights'
        
    # calculates rewards from agent's current state (call to when taking a step)
    def reward(self, state):
        total_reward = 0
        for idx, reward in enumerate(self._rewards):
            value = reward.reward(state)
            state['reward_from_' + reward._name] = value
            total_reward += self.reward_weights[idx] * value
        nSteps = state['nSteps']
        normalized_steps = nSteps / self.max_steps 
        decayed_reward = total_reward * math.exp(self.decay_factor*normalized_steps)
        state['total_reward'] = decayed_reward
        return decayed_reward