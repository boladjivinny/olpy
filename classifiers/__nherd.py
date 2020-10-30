import numpy as np

from olpy._model import Model, BCModelWithLabelEncoding

class NormalHerd(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y, a=1, C=1):
        super().__init__(X, y)
        self.Sigma  = a * np.eye(X.shape[1])
        self.gamma  = 1/C

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            v_t     = x_t @ self.Sigma @ x_t.T
            m_t     = y_t * f_t
            l_t     = 1 - m_t
            if l_t > 0:
                beta_t      = 1 / (v_t + self.gamma)
                alpha_t     = max(0, 1-m_t) * beta_t
                S_x_t       = x_t @ self.Sigma.T
                self.weights= self.weights + alpha_t
                self.Sigma  = self.Sigma - beta_t ** 2 * (
                                v_t + 2 * self.gamma) @ S_x_t.T @ S_x_t
        return hat_y_t