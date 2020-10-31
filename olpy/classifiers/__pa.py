import numpy as np
from numpy import linalg as LA
import math

from sklearn.metrics import zero_one_loss, log_loss, mean_squared_error
from scipy.stats import invgauss
from olpy._model import Model, BCModelWithLabelEncoding

class PassiveAggressive(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y):
        super().__init__(X, y)

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            l_t     = max(0, 1 - y_t * f_t)
            if l_t > 0:
                s_t     = LA.norm(x_t) ** 2
                gamma_t = self.__get_gamma(l_t, s_t)
                self.weights    = self.weights + gamma_t * y_t * x_t
        
        return hat_y_t
    
    def __get_gamma(self, l_t, s_t):
        return l_t / s_t if s_t > 0 else 1


class PassiveAggressiveI(PassiveAggressive):
    def __init__(self, X, y, C):
        super().__init__(X, y)
        self.C = C
    def __get_gamma(self, l_t, s_t):
        return min(self.C, l_t / s_t)


class PassiveAggressiveII(PassiveAggressive):
    def __init__(self, X, y, C):
        super().__init__(X, y)
        self.C = C

    def __get_gamma(self, l_t, s_t):
        return l_t / (s_t + (1/(2 * self.C )))
