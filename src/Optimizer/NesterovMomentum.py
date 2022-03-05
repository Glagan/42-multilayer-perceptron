
import numpy as np


class NesterovMomentum:
    def __init__(self, momentum=0.9, v=0.5) -> None:
        self.momentum = momentum
        self.v = v

    def optimize(self, d_weights: np.ndarray, learning_rate: float, _: int) -> np.ndarray:
        v_previous = self.v
        self.v = self.momentum * self.v - learning_rate * d_weights
        return - self.momentum * (v_previous + (1. + self.momentum)) * self.v
