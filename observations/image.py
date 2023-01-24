# used to handle image observations saved as np arrays
from observations.observation import Observation

class Image(Observation):

    # constructor
    def __init__(self, _data, data_path=None, is_gray=False):
        super().__init__(_data=_data, data_path=data_path)
        self.is_gray = is_gray