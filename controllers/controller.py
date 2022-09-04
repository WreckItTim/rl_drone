# abstract class used to handle all components
from component import Component

class Controller(Component):

	# constructor
	def __init__(self):
		# tell the init_wrapper to not add to configuration file as component or create a unique name
		self._add_timers = False
		self._set_name = False
		self._add_to_configuration = False

	# runs control on components
	def run(self):
		raise NotImplementedError

	def connect(self):
		super().connect()