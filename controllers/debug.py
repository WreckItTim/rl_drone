from controllers.controller import Controller
from component import _init_wrapper
from configuration import Configuration

# will load at a list of all components
# allows user to view and debug each component independently 
# or move drone around
class Debug(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, drone_component):
		super().__init__()

	# runs control on components
	def run(self):
		configuration = Configuration.get_active()
		component_names = list(configuration.components.keys())
		while(True):
			print('Enter component _name or index to debug, list to see components, or reset')
			user_input = input().lower()
			if user_input == 'quit':
				break
			elif user_input == 'list':
				component_names = list(configuration.components.keys())
				for idx, component_name in enumerate(component_names):
					print(idx, ':', component_name)
			elif user_input == 'reset':
				self._drone.reset()
			elif user_input == 'move':
				user_input = input()
				x, y, z, yaw = [float(_) for _ in user_input.split(' ')]
				self._drone.teleport(x, y, z, yaw)
			else:
				if user_input in component_names: 
					configuration.get_component(user_input).debug()
				else:
					try:
						index = int(user_input)
						if index <= 0 or index >= len(component_names):
							print('invalid entry')
							continue
					except ValueError:
						print('invalid entry')
						continue
					component_name = component_names[index]
					configuration.get_component(component_name).debug()