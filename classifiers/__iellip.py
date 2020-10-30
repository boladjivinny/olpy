import numpy as np
import math

from olpy._model import Model, BCModelWithLabelEncoding

class ImprovedEllipsoid(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y, a=1, b=1, c=1):
        super().__init__(X, y)
        self.Sigma  = a * np.eye(X.shape[1])
        self.b      = b
        self.c      = c

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            l_t     = (hat_y_t != y_t)
            v_t     = x_t @ self.Sigma @ x_t.T
            m_t     = y_t * f_t
            if l_t:
                if v_t != 0:
                    alpha_t         = (1 - m_t) / math.sqrt(v_t)
                    g_t             = y_t * x_t / math.sqrt(v_t)
                    S_x_t           = g_t @ self.Sigma.T
                    self.weights   += alpha_t * S_x_t
                    self.Sigma      = (self.Sigma - self.c @ S_x_t.T @ S_x_t) / (1 - self.c)
            self.c  *= self.b
        return hat_y_t