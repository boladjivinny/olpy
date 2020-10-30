import numpy as np
from olpy._model import Model, BCModelWithLabelEncoding

class AROW(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y, r=1, a=1):
        super().__init__(X, y)
        self.Sigma = a * np.eye(X.shape[1])
        self.r = r

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            m_t = f_t
            v_t = x_t @ self.Sigma @ x_t.T
            l_t = max(0, 1-m_t*y_t)
            if l_t > 0:
                beta_t = 1/(v_t + self.r)
                alpha_t = l_t * beta_t
                S_x_t = x_t * self.Sigma
                self.weights += alpha_t * y_t * S_x_t
                self.Sigma -= beta_t * S_x_t.T @ S_x_t
        return hat_y_t