import numpy as np
import math

from scipy.stats import norm
from olpy._model import OnlineLearningModel


class ConfidenceWeighted(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, eta=0.7, a=1):
        super().__init__()
        self.a = a
        self.phi = norm.ppf(eta)
        self.psi = 1 + (self.phi ** 2) / 2
        self.xi = 1 + self.phi ** 2

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        # Added the absolute value function to avoid math errors
        v_t = x @ self.Sigma @ x.T
        m_t = y * f_t
        l_t = self.phi * math.sqrt(v_t) - m_t
        if l_t > 0:
            alpha_t = self.get_alpha(m_t, v_t)
            u_t = 0.25 * (-alpha_t * v_t * self.phi + math.sqrt(
                alpha_t ** 2 * v_t ** 2 * self.phi ** 2 + 4 * v_t)) ** 2
            beta_t = alpha_t * self.phi / (math.sqrt(u_t) +
                                           alpha_t * self.phi * v_t)
            S_x_t = np.expand_dims(x @ self.Sigma, axis=0)
            self.weights += alpha_t * y * np.squeeze(S_x_t)
            self.Sigma -= beta_t * S_x_t.T @ S_x_t

    def get_alpha(self, m_t, v_t):
        return max(0, (-m_t * self.psi + math.sqrt((m_t ** 2 *
                                                    self.phi ** 4) / 4 + v_t * self.phi ** 2 *
                                                   self.xi)) / (v_t * self.xi))

    def setup(self, X: np.ndarray, Y: np.ndarray):
        self.Sigma = self.a * np.eye(X.shape[1])


class SoftConfidenceWeighted(ConfidenceWeighted):
    def __init__(self, eta=0.75, a=1, C=1):
        super().__init__(eta=eta, a=a)
        self.C = C

    def get_alpha(self, m_t, v_t):
        alpha_t = max(0, (-m_t * self.psi + math.sqrt((m_t ** 2 *
                                                       self.phi ** 4) / 4 + v_t * self.phi ** 2 *
                                                      self.xi)) / (v_t * self.xi))
        return min(alpha_t, self.C)
