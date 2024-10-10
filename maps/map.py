# abstract class used to handle maps - where the drone is flying in
# this is the part that connects the program to the real application
from component import Component


class Map(Component):
	# constructor
	def __init__(self):
		pass

	def connect(self):
		super().connect()