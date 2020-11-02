import numpy as np
from numpy import linalg as LA

from olpy._model import OnlineLearningModel


class PassiveAggressive(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self):
        super().__init__()

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        l_t = max(0, 1 - y * f_t)
        if l_t > 0:
            s_t = LA.norm(x) ** 2
            gamma_t = self.get_gamma(l_t, s_t)
            self.weights = self.weights + gamma_t * y * x

    def get_gamma(self, l_t, s_t):
        return l_t / s_t if s_t > 0 else 1


class PassiveAggressiveI(PassiveAggressive):
    def __init__(self, C=1):
        super().__init__()
        self.C = C

    def get_gamma(self, l_t, s_t):
        return min(self.C, l_t / s_t)


class PassiveAggressiveII(PassiveAggressive):
    def __init__(self, C=1):
        super().__init__()
        self.C = C

    def get_gamma(self, l_t, s_t):
        return l_t / (s_t + (1/(2 * self.C )))
