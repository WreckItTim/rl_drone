# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from component import _init_wrapper
from sensors.sensor import Sensor
from observations.image import Image
from transformers.transformer import Transformer
import numpy as np

class MultiStack(Observer):
    
    # constructor
    @_init_wrapper
    def __init__(
        self, 
        observers_components, 
		output_height, 
		output_width,
        stack='v',
    ):
        super().__init__(
        )
        self._output_shape = (output_height, output_width, 1)
        
    # gets observations
    def observe(self):
        datas = []
        for observer in self._observers:
            datas.append(observer.observe().to_stack(observer.output_height, observer.output_width))
        data = None
        if self.stack == 'v':
            data = np.vstack(datas)
        elif self.stack == 'h':
            data = np.hstack(datas)
        stacked_image = Image(
            _data=data,
            is_gray=True,
        )
        return stacked_image

    def reset(self):
        for observer in self._observers:
            observer.reset()