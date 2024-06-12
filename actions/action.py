from component import Component

# abstract class used to handle an individual action
class Action(Component):

	# contstructor
	def __init__(self):
		raise NotImplementedError
	
	# establish connection to be used in episode - connects all components to eachother and calls child connect() for anything else needed
	# WARNING: if you overwrite this make sure to call super()
	def connect(self, state=None):
		super().connect()

	def debug(self, state=None):
		print('enter rl_output:')
		user_input = input().lower()
		try:
			rl_out = float(user_input)
			state = {
				'rl_output' : [rl_out] * (self._idx + 1)
			}
			self.step(state)
		except ValueError:
			print('invalid entry')
		print('collided?', self._drone.check_collision())