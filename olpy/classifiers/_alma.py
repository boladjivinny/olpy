import numpy as np
import math

from numpy import linalg as LA

from olpy import OnlineLearningModel


class ALMA(OnlineLearningModel):
    def __init__(self, p=2, **kwargs):
        super().__init__()

        self.p = p
        self.alpha = kwargs.get('eta', 1.0)
        self.B = 1 / self.alpha
        self.C = kwargs.get('C', 1)
        self.q = kwargs.get('q', p)
        self.k = 1

        print(self.C, self.q, self.p, self.alpha)

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        gamma_k = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)
        l_t = (1 - self.alpha) * gamma_k - y * f_t
        if l_t > 0:
            eta_k = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))
            self.weights = self.weights + eta_k * y * x
            norm_w = LA.norm(self.weights, ord=self.p)
            self.weights = self.weights / (max(1, norm_w))
            self.k += 1

    def __f_mapping(self):
        return (np.sign(self.weights) * np.abs(self.weights) ** (self.q - 1)) / (
                    LA.norm(self.weights, ord=self.q) ** (self.q - 2))

    def __f_inv(self, theta):
        return (np.sign(theta) * np.abs(theta) ** (self.p - 1)) / (
                    LA.norm(theta, self.p) ** (self.p - 2))