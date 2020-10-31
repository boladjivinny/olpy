import numpy as np
import math

from numpy import linalg as LA

from olpy._model import Model
from olpy.preprocessing._labels import LabelEncoder

class ALMA(Model):
    def __init__(self, X, y, p=2, **kwargs):
        super().__init__(X, y)

        positive_label = kwargs.get('positive_label', 1)
        self.y, self.labels = LabelEncoder(self.y, positive_label=positive_label).fit_transform(return_labels=True)

        self.p = p
        self.alpha = kwargs.get('alpha', 1.0)
        self.B = kwargs.get('B', 1)
        self.C = kwargs.get('C', 1)
        self.q = kwargs.get('q', p)
        
        self.weights = np.zeros(self.X.shape[1])
        self.k = 1

    def fit(self):
        # We loop through the whole dataset
        for t in self.X.shape[0]:

            psi_k = self.B * math.sqrt(self.k) * math.sqrt(self.p - 1) / self.k
            x_hat_t = self.X[t, :] / LA.norm(self.X[t, :], ord=self.p)

            if self.y[t] * np.dot(self.weights, x_hat_t) <= (1 - self.alpha) * psi_k:
                mu_k = self.C / (math.sqrt(self.p - 1)*math.sqrt(self.k))
                _w_k_prime = self.__f_inv(self.__f_mapping() + (mu_k * self.y[t] * x_hat_t))
            self.k += 1

    def predict(self, data):
        #with pred = data @ self.weights:
        return [self.labels[0] if val <=0 else 1 for val in data @ self.weights ]

    def __f_mapping(self):
        return (np.sign(self.weights) * np.abs(self.weights) ** (self.q - 1)) / (
                    LA.norm(self.weights, ord=self.q) ** (self.q - 2))

    def __f_inv(self, theta):
        return (np.sign(theta) * np.abs(theta) ** (self.p - 1)) / (
                    LA.norm(theta, self.p) ** (self.p - 2))