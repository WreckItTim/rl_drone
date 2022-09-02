# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor

from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer

class Single(Observer):
    
    # constructor
    @_init_wrapper
    def __init__(
        self, 
        sensor_component, 
		output_height, 
		output_width,
        transformers_components=[],
    ):
        super().__init__(
        )
        self._output_shape = (output_height, output_width, 1)
        
    # gets observations
    def observe(self):
        observation = self._sensor.sense()
        for transformer in self._transformers:
            transformer.transform(observation)
        return observation

    def reset(self):
        super().reset()
        self._sensor.reset()