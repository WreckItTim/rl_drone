from others.other import Other
from component import _init_wrapper

# curriculum learning changes goal distance (r) every few episodes
class Curriculum(Other):

	@_init_wrapper
	def __init__(self, 
			goal_component,
			model_component,
			start_r,
			intervals, # progressive list of when to advance to next level
			distances, # progressive list of goal distances
			final_form = False,
			amp_idx = 0,
		):
		pass

	def reset_learning(self, state=None):
		self._goal.mean_r = self.start_r
		self.final_form = False
		self.amp_idx = 0

	def update(self):
		if self.final_form:
			return
		amp_interval = self.amp_intervals[self.amp_idx]
		if self._model.nEpisodes > amp_interval:
			self._goal.mean_r = self.amp_distances[self.amp_idx]
			self.amp_idx += 1
			if self.amp_idx >= len(self.amp_intervals):
				self._goal.std_r =  (self._goal.mean_r - self.start_r)/2
				self.final_form = True
