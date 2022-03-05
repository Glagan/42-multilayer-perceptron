
import numpy as np


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=0.00000001) -> None:
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(self, d_weights: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        self.m = self.beta1 * self.m + (1-self.beta1) * d_weights
        mt = self.m / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (d_weights ** 2)
        vt = self.v / (1-self.beta2 ** self.t)
        return - learning_rate * mt / (np.sqrt(vt) + epoch)
