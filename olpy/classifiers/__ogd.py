import numpy as np
import math

from sklearn.metrics import zero_one_loss, log_loss, mean_squared_error
from olpy._model import Model, BCModelWithLabelEncoding

class OnlineGradientDescent(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y, C=1, loss_function=zero_one_loss, **kwargs):
        super().__init__(X, y)
        self.eta            = C
        self.loss_function  = loss_function
        self.t              = kwargs.get('t', 1)

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            eta_t   = self.eta / math.sqrt(self.t)
            l_t     = self.loss_function(y_t, hat_y_t)
            if l_t > 0:
                if self.loss_function == log_loss:
                    self.weights    = self.weights + eta_t * y_t * x_t * ( 1 / (
                                            1 + math.exp(y_t * f_t)))
                elif self.loss_function == mean_squared_error:
                    self.weights    = self.weights - eta_t * (f_t - y_t) * x_t
                else:
                    self.weights    = self.weights + eta_t * y_t * x_t

            self.t += 1
        
        return hat_y_t