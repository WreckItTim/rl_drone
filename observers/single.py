# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor

from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer

class Single(Observer):
    # constructor
    @_init_wrapper
    def __init__(self, sensor_name='', transformer_names=[], name=None, please_write=False, write_directory=None):
        super().__init__(please_write=please_write, write_directory=write_directory)
        
    # gets observations than transcribes for input into RL model
    def observe(self):
        observation = self._sensor.sense()
        for transformer in self._transformers:
            transformer.transform(observation)
        return super().observe(observation)

    # tests this component
    def test(self):
        print('making observation...')
        observation = self._sensor.sense()
        observation.write('temp/', 'observation')
        for transformer in self._transformers:
            print(f'transforming observation with {transformer._child().__name__}...')
            transformer.transform(observation)
            observation.write('temp/', transformer._child().__name__ + '_observation')
        print('displaying observation...')
        observation.display()