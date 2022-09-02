# Single modality that concats 2 observations - most recent and one n-frames behind
from observers.observer import Observer
from sensors.sensor import Sensor
from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer
import numpy as np

class SingleLag(Observer):
    
    # constructor
    @_init_wrapper
    def __init__(self, 
                 sensor_component, 
                 output_height, 
                 output_width, 
                 n_frames_lag, 
                 transformers_components=[],
                 ):
        super().__init__()
        self._output_shape = (output_height, 2*output_width, 1)
        self._blank = np.zeros(
            (output_height, output_width,1), 
            dtype=np.uint8,
            )
        self._queue = [self._blank] * n_frames_lag
        self._rotating_index
        
    def update_queue(self, data):
        self._queue[self._rotating_index] = data
        self._rotating_index += 1
        if self._rotating_index >= self.n_frames_lag:
            self._rotating_index = 0

    # gets observations 
    def observe(self):
        new_observation = self._sensor.sense()
        for transformer in self._transformers:
            transformer.transform(new_observation)
        data = np.hstack([self._queue[self._rotating_index], new_observation._data])
        observation = self._sensor.create_observation(data)
        self.update_queue(new_observation._data)
        return observation

    def reset(self):
        super().reset()
        self._sensor.reset()
        self._queue = [self._blank] * self.n_frames_lag