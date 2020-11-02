import numpy as np
from olpy._model import OnlineLearningModel


class AROW(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, r=1, a=1):
        super().__init__()
        self.a = a
        self.r = r
        self.Sigma = None

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        v_t = x @ self.Sigma @ x.T
        l_t = max(0, 1 - f_t * y)
        print(f_t, l_t, v_t)
        if l_t > 0:
            beta_t = 1 / (v_t + self.r)
            alpha_t = l_t * beta_t
            S_x_t = np.expand_dims(x @ self.Sigma.T, axis=0)
            self.weights += alpha_t * y * np.squeeze(S_x_t)
            self.Sigma -= beta_t * S_x_t.T @ S_x_t

    def setup(self, X: np.ndarray, Y: np.ndarray):
        self.Sigma = self.a * np.eye(X.shape[1])
