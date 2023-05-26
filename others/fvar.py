from others.other import Other
from component import _init_wrapper
import pickle
import time

# safe write to file (shared by multiple processes)
class FVar(Other):

	@_init_wrapper
	def __init__(self, 
			fpath,
			default,
		):
		pass

	def reset_learning(self, state=None):
		self.speak(self.default)

	def listen(self):
		waiting = True
		while waiting:
			try:
				with open(self.fpath, 'rb') as f:
					value = pickle.load(f) 
				with open(self.fpath, 'wb') as f:
					pickle.dump(self.default, f) 
				waiting = False
			except IOError:
				time.sleep(.1)

	def speak(self, value):
		waiting = True
		while waiting:
			try:
				with open(self.fpath, 'wb') as f:
					pickle.dump(value, f) 
				waiting = False
			except IOError:
				time.sleep(.1)