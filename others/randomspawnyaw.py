# abstract class used to handle abstract components
from others.other import Other
from component import _init_wrapper
import random

# randomly set drone rotation
class RandomSpawnYaw(Other):

    @_init_wrapper
    def __init__(self, drone_component, environment_component, yaw_min, yaw_max):
        pass

    def reset(self):
        if not self._environment._evaluating:
            degrees = random.uniform(self.yaw_min, self.yaw_max)
            self._drone.set_yaw(degrees)

    def activate(self):
        self.reset()