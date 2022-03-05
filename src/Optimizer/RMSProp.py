
import numpy as np


class RMSProp:
    def __init__(self, decay_rate=0.9) -> None:
        self.decay_rate = decay_rate
        self.cache = 1

    def optimize(self, d_weights: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        cache = self.decay_rate * cache + \
            (1 - self.decay_rate) * d_weights ** 2
        return - learning_rate * d_weights / (np.sqrt(cache) + epoch)
