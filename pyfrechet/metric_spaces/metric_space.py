from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np
from .utils import coalesce_weights

class MetricSpace(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _d(self, x, y) -> float:
        pass

    def d(self, x, y) -> Union[float, np.ndarray]:

        # Check if arguments are instances of the MetricData class
        x_is_md = type(x).__name__ == 'MetricData'
        y_is_md = type(y).__name__ == 'MetricData'

        if not (x_is_md or y_is_md):
            return self._d(x, y)
        elif x_is_md and y_is_md:
            # If both are MetricData instances it computes the distance between them
            # provided that lengths agree
            assert len(x) == len(y), "Lenghts of x and y differ"
            return np.array([ self._d(x[i], y[i]) for i in range(len(x)) ])
        elif y_is_md:
            return np.array([ self._d(x, y[i]) for i in range(len(y)) ])
        else:
            return self.d(y, x)

    @abstractmethod
    def _frechet_mean(self, y, w=None):
        pass

    def frechet_mean(self, y, w=None):
        w = coalesce_weights(w, y)
        return self._frechet_mean(y, w)
    
    def frechet_var(self, y, w=None):
        w = coalesce_weights(w, y)
        y_bar = self.frechet_mean(y, w=w)
        return np.sum([ w[i] * self._d(y_bar, self.index(y, i))**2 for i in range(y.shape[0]) ])

    def index(self, y, i):
        return y[i, :]