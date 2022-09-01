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
		output_width,
		output_height, 
        transformers_components=[], 
        please_write=False, 
        write_directory=None,
    ):
        super().__init__(
            please_write=please_write, 
            write_directory=write_directory,
        )
        self._output_shape = (output_width, output_height, 1)
        
    # gets observations
    def observe(self):
        observation = self._sensor.sense()
        for transformer in self._transformers:
            transformer.transform(observation)
        return observation

    def reset(self):
        super().reset()
        self._sensor.reset()