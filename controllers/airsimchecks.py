from controllers.controller import Controller
from component import _init_wrapper
from configuration import Configuration
import random
import rl_utils as utils
import pickle

# takes series of random actions
# measures position and orientation before and after each action
class AirSimChecks(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, drone_component, actor_component):
		super().__init__()

	# runs control on components
	def run(self):
		nRuns = 1_000_000
		nIters = 100
		results = {}
		part = 1
		for r in range(nRuns):
			if r > 0 and r % 1000 == 0:
				results = {}
				part += 1
			utils.speak(f'on run {r} ...')
			results[r] = {}
			self._drone._airsim._client.reset()
			self._drone._airsim._client.enableApiControl(True)
			self._drone._airsim._client.armDisarm(True)
			self._drone._airsim._client.takeoff()
			for i in range(nIters):
				action = random.choice(self._actor._actions)
				rl_output = random.uniform(action.min_space, action.max_space)
				state = {
					'rl_output' : [rl_output] * (action._idx + 1)
				}
				pos_before = self._drone.get_position()
				yaw_before = self._drone.get_yaw()
				output = action.step(state)
				collided = self._drone.check_collision()
				pos_after = self._drone.get_position()
				yaw_after = self._drone.get_yaw()
				results[r][i] = {
					'action':action._name,
					'rl_output':rl_output,
					'pos_before':pos_before,
					'yaw_before':yaw_before,
					'output':output,
					'collided':collided,
					'pos_after':pos_after,
					'yaw_after':yaw_after,
				}
				if collided:
					break
			pickle.dump(results, open(utils.get_global_parameter('working_directory') + 'results_part' + str(part) + '.p', 'wb'))
