import numpy as np

from olpy import OnlineLearningModel


class NormalHerd(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, a=1, C=1):
        super().__init__()
        self.a = a
        self.gamma = 1/C
        self.Sigma = None

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        v_t = x @ self.Sigma @ x.T
        m_t = y * f_t
        l_t = 1 - m_t
        if l_t > 0:
            beta_t = 1 / (v_t + self.gamma)
            alpha_t = max(0, 1 - m_t) * beta_t
            S_x_t = np.expand_dims(x @ self.Sigma.T, axis=0)
            self.weights = self.weights + alpha_t * y * np.squeeze(S_x_t)
            self.Sigma = self.Sigma - (beta_t ** 2) * (v_t + 2 * self.gamma) * S_x_t.T @ S_x_t

    def setup(self, X: np.ndarray, Y: np.ndarray):
        self.Sigma = self.a * np.eye(X.shape[1])