# penalizes colliding with objects
from rewards.reward import Reward
from component import _init_wrapper

class Avoid(Reward):
    # constructor
    @_init_wrapper
    def __init__(self, drone_name='', name=None):
        super().__init__()

    # -1 for a collision, +1 for dodging collision
    def evaluate(self, state):
        if 'has_collided' not in state:
            state['has_collided'] = self._drone.check_collision()
        has_collided = state['has_collided']
        total_reward = 1
        if has_collided:
            total_reward = -1
        return total_reward

    def test(self):
        print(f'reward before collision:{self.evaluate({})}')
        print('attempting collision...')
        self._drone.move(40, 0, 0, 4, front_facing=False)
        print(f'reward after collision:{self.evaluate({})}')
