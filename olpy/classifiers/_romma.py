import numpy as np
from numpy import linalg as LA

from olpy._model import OnlineLearningModel


class ROMMA(OnlineLearningModel):
    def __init__(self):
        super().__init__()

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        if self.is_mistake(x, y):
            if LA.norm(self.weights) == 0:
                self.weights += y * x
            else:
                deno = (LA.norm(x) * LA.norm(self.weights)) ** 2 - f_t ** 2
                if deno != 0:
                    coe_1 = ((LA.norm(x) * LA.norm(self.weights)) ** 2 - y * f_t) / deno
                    coe_2 = (LA.norm(self.weights) ** 2 * (y - f_t)) / deno
                    self.weights = coe_1 * self.weights + coe_2 * x

    def is_mistake(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        return hat_y_t != y


class AgressiveROMMA(ROMMA):
    def is_mistake(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        l_t = 1 - y * f_t
        return l_t > 0
