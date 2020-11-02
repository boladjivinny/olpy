import numpy as np
from olpy._model import Model, OnlineLearningModel


class NAROW(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, a=1, b=1):
        super().__init__()
        self.a = a
        self.b = b

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        v_t = x @ self.Sigma @ x.T
        m_t = f_t * y
        l_t = 1 - m_t
        if l_t > 0:
            chi_t = x @ self.Sigma @ x.T
            if chi_t > (1 / self.b):
                r_t = chi_t / (self.b * chi_t - 1)
            else:
                r_t = np.NINF
            beta_t = 1 / (v_t + r_t)
            alpha_t = max(0, 1 - m_t) * beta_t
            S_x_t = np.expand_dims(x @ self.Sigma.T, axis=0)
            self.weights += alpha_t * y * np.squeeze(S_x_t)
            self.Sigma -= beta_t * S_x_t.T @ S_x_t

    def setup(self, X: np.ndarray, Y: np.ndarray):
        self.Sigma = self.a * np.eye(X.shape[1])
