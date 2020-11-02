import numpy as np
import math

from sklearn.metrics import zero_one_loss, log_loss, mean_squared_error, hinge_loss
from olpy import OnlineLearningModel


class OnlineGradientDescent(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, C=1, loss_function=zero_one_loss, **kwargs):
        super().__init__()
        self.eta = C
        self.loss_function = loss_function
        self.t = kwargs.get('t', 1)

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        eta_t = self.eta / math.sqrt(self.t)

        # Changed the parameters to call the loss function as it seems they
        # expect at least two values
        if self.loss_function == hinge_loss:
            l_t = self.loss_function([y, -y], [f_t, -f_t])
        else:
            l_t = self.loss_function([y], [hat_y_t])

        if l_t > 0:
            if self.loss_function == log_loss:
                self.weights = self.weights + eta_t * y * x * (1 / (1 + math.exp(y * f_t)))
            elif self.loss_function == mean_squared_error:
                self.weights = self.weights - eta_t * (f_t - y) * x
            else:
                self.weights = self.weights + eta_t * y * x

        self.t += 1
