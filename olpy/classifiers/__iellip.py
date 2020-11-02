import numpy as np
import math

from olpy import OnlineLearningModel


class ImprovedEllipsoid(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, a=1, b=0.3, c=0.1):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.Sigma = None

    def update(self, x, y):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        l_t = (hat_y_t != y)
        v_t = x @ self.Sigma @ x.T
        m_t = y * f_t
        if l_t:
            if v_t != 0:
                print(v_t)
                alpha_t = (1 - m_t) / math.sqrt(v_t)
                g_t = y * x / math.sqrt(v_t)
                # Added an axis to avoid errors
                S_x_t = np.expand_dims(g_t @ self.Sigma.T, axis=0)
                self.weights = self.weights + alpha_t * np.squeeze(S_x_t)
                self.Sigma = (self.Sigma - self.c * (S_x_t.T @ S_x_t)) / (1 - self.c)
        self.c *= self.b

    def setup(self, X: np.ndarray, Y: np.ndarray):
        self.Sigma = self.a * np.eye(X.shape[1])
