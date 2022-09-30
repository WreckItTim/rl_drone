# abstract class used to handle all components
from controllers.controller import Controller
from component import _init_wrapper
from configuration import Configuration

class Empty(Controller):
	# constructor
	@_init_wrapper
	def __init__(self):
		super().__init__()

	# runs control on components
	def run(self):
		pass