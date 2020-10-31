import numpy as np
from numpy import linalg as LA

from olpy._model import Model, BCModelWithLabelEncoding
from olpy.preprocessing._labels import LabelEncoder


class ROMMA(BCModelWithLabelEncoding):
    def __init__(self, X, y):
        super().__init__(X, y)

    def fit(self):
        for t in self.X.shape[0]:
            f_t = self.weights.dot(self.X[t])
            hat_y_t = 1 if f_t >= 0 else -1
            if hat_y_t != self.y[t]:
                if LA.norm(self.weights) == 0:
                    self.weights += self.y[t] * self.X[t]
                else:
                    deno = (LA.norm(self.X[t]) * LA.norm(self.weights))**2 - f_t ** 2
                    if deno != 0:
                        coe_1 = ((LA.norm(self.X[t]) * LA.norm(self.weights))**2 - self.y[t] * f_t) / deno
                        coe_2 = (LA.norm(self.weights) ** 2 * (self.y[t] - f_t)) / deno
                        self.weights = coe_1*self.weights + coe_2 * self.X[t]


class AgressiveROMMA(ROMMA):
    def fit(self):
        for t in self.X.shape[0]:
            f_t = self.weights.dot(self.X[t])
            l_t = 1 - y_t * f_t
            if l_t > 0:
                if LA.norm(self.weights) == 0:
                    self.weights += self.y[t] * self.X[t]
                else:
                    deno = (LA.norm(self.X[t]) * LA.norm(self.weights))**2 - f_t ** 2
                    if deno != 0:
                        coe_1 = ((LA.norm(self.X[t]) * LA.norm(self.weights))**2 - self.y[t] * f_t) / deno
                        coe_2 = (LA.norm(self.weights) ** 2 * (self.y[t] - f_t)) / deno
                        self.weights = coe_1*self.weights + coe_2 * self.X[t]
