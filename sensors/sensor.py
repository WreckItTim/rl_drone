# abstract class used to handle sensors
from component import Component
import utils

# calls self.sense2() from child class...
	# must set sense2() from child and return the raw observation
	# parent sense() will run pre/post processing on raw observation
# if raw_code is set to not None then..
	# raw data fetches are temporarily logged
	# when sensors with same raw_code call parent sense()..
		# will check if raw observation has been logged
		# this is done to avoid multiple raw data fetches
class Sensor(Component):
	# constructor, offline is a boolean that handles if sensor goes offline
	def __init__(self,
			  offline,
			  raw_code=None,
			  # raw code is so same sensor wont fetch twice 
			  # #(can give any json-transcirbable code other than None)
			  ):
		pass

	# creates an observation object given a data array
	def create_obj(self, data):
		raise NotImplementedError

	def presense(self):
		if self.raw_code is not None:
			#print('fetch raw code', self.raw_code)
			data = utils.get_global_parameter('raw_' + self.raw_code)
			if data is None:
				return None
			raw_observation = self.create_obj(data)
			return raw_observation
		return None

	def postsense(self, observation):
		if self.raw_code is not None:
			#print('save raw code', self.raw_code)
			utils.set_global_parameter('raw_' + self.raw_code, observation.to_numpy())
		transformed = self.transform(observation)
		return transformed

	# after all data fetches for this step
	def cleanup(self):
		if self.raw_code is not None:
			#print('del raw code', self.raw_code)
			utils.del_global_parameter('raw_' + self.raw_code)
	
	# fetch a response from sensor
	def sense(self):
		#print('sense', self._child())
		raw_observation = self.presense()
		if raw_observation is None:
			#print('create raw code', self.raw_code)
			raw_observation = self.sense2()
		#print('raw', raw_observation.to_numpy())
		observation = self.postsense(raw_observation)
		#print('transformed', observation.to_numpy())
		#x = input()
		return observation

	def transform(self, observation):
		if self._transformers is not None:
			for transformer in self._transformers:
				observation_conversion = transformer.transform(observation)
				if observation_conversion is not None:
					observation = observation_conversion
		return observation

	# when using the debug controller
	def debug(self):
		observation = self.sense()
		observation.display()

	def connect(self):
		super().connect()
