# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor

from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer

class Single(Observer):
    # constructor
    @_init_wrapper
    def __init__(self, sensor_component, output_shape, transformers_components=[], please_write=False, write_directory=None):
        super().__init__(please_write=please_write, write_directory=write_directory)
        
    # gets observations than transcribes for input into RL model
    def observe(self):
        observation = self._sensor.sense()
        for transformer in self._transformers:
            transformer.transform(observation)
        return observation