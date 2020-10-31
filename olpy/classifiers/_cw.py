import numpy as np
import math

from scipy.stats import invgauss
from olpy._model import Model, BCModelWithLabelEncoding

class ConfidenceWeighted(BCModelWithLabelEncoding):
    # Need to check the r parameter 
    def __init__(self, X, y, eta=1, a=1):
        super().__init__(X, y)
        self.Sigma  = a * np.eye(X.shape[1])
        self.phi    = invgauss.cdf(eta, 0)
        self.psi    = 1 + (self.phi ** 2) / 2
        self.xi     = 1 + self.phi ** 2

    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            v_t     = x_t @ self.Sigma @ x_t.T
            m_t     = y_t * f_t
            l_t     = self.phi * math.sqrt(v_t) - m_t
            if l_t > 0:
                alpha_t         = self.__get_alpha(m_t, v_t)
                u_t             = 0.25 * (-alpha_t * v_t * self.phi + math.sqrt(
                                        alpha_t ** 2 * v_t ** 2 * self.phi ** 2+ 4 * v_t )) ** 2
                beta_t          = alpha_t * self.phi/ (math.sqrt(u_t) + 
                                        alpha_t * self.phi * v_t)
                S_x_t           = x_t @ self.Sigma
                self.weights   += alpha_t * y_t * S_x_t
                self.Sigma     -= beta_t * S_x_t.T @ S_x_t
        return hat_y_t

    def __get_alpha(self, m_t, v_t):
        return max(0,(-m_t * self.psi + math.sqrt((m_t ** 2 * 
                            self.phi ** 4) / 4 + v_t * self.phi ** 2 * 
                            self.xi)) / (v_t * self.xi))


class SoftConfidenceWeighted(ConfidenceWeighted):
    def __init__(self, X, y, eta=1, a=1, C=1):
        super().__init__(X, y, eta=eta, a=a)
        self.C  = C

    def __get_alpha(self, m_t, v_t):
        alpha_t = max(0,(-m_t * self.psi + math.sqrt((m_t ** 2 * 
                            self.phi ** 4) / 4 + v_t * self.phi ** 2 * 
                            self.xi)) / (v_t * self.xi))
        return min(alpha_t, self.C)