import numpy as np
import math

from scipy.stats import norm
from olpy._model import OnlineLearningModel


class CW(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, eta=0.7, a=1, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiate a Confidence Weighted model for training.
        
        Dredze, M.; Crammer, K. & Pereira, F.
        Confidence-Weighted linear classification 
        Proc. 25th Int. Conf. Machine Learning, 
        Association for Computing Machinery, 2008, 264-271

        Parameters
        ----------
        a   : float, default 1
            Initial variance parameter, a > 0
        eta : float, default=0.7
            Value to which the inverse gaussian PDF is applied
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)
        self.a = a
        self.eta = eta

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        v_t = x @ np.diag(np.diag(self.sigma)) @ x.T
        m_t = y * decision
        loss = self.phi * math.sqrt(v_t) - m_t
        if loss > 0:
            alpha_t = self._get_alpha(m_t, v_t)
            u_t = 0.25 * (-alpha_t * v_t * self.phi + math.sqrt(
                alpha_t ** 2 * v_t ** 2 * self.phi ** 2 + 4 * v_t)) ** 2
            beta_t = alpha_t * self.phi / (math.sqrt(u_t) +
                                           alpha_t * self.phi * v_t)
            sigma = np.expand_dims(x @ self.sigma, axis=0)
            self.weights += alpha_t * y * np.squeeze(sigma)
            self.sigma -= beta_t * sigma.T @ sigma

    def _get_alpha(self, m_t, v_t):
        """Computes the alpha for the CW/SCW algorithm"""
        return max(0, (-m_t * self.psi + math.sqrt((m_t ** 2 *
                                                    self.phi ** 4) / 4 + v_t * self.phi ** 2 *
                                                   self.xi)) / (v_t * self.xi))

    def _setup(self, X: np.ndarray):
        self.sigma = self.a * np.eye(X.shape[1])
        self.phi = norm.ppf(self.eta)
        self.psi = 1 + (self.phi ** 2) / 2
        self.xi = 1 + self.phi ** 2

    def get_params(self):
        return {'a': self.a, 'eta': self.eta, 'num_iterations': self.num_iterations}


class SCW(CW):
    def __init__(self, eta=0.7, C=1, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiate a SOft Confidence Weighted model for training.
        
        Wang, J.; Zhao, P. & Hoi, S. C. H.
        Exact Soft Confidence-Weighted learning 
        CoRR, 2012, abs/1206.4612

        Parameters
        ----------
        C   : float, default 1
            SCW parameter, C > 0
        eta : float, default=0.7
            Value to which the inverse gaussian PDF is applied
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.

        Returns
        -------
        None
        """
        super().__init__(eta=eta, a=1, num_iterations=num_iterations,\
             random_state=random_state, positive_label=positive_label)
        self.C = C

    def _get_alpha(self, m_t, v_t):
        alpha_t = max(0, (-m_t * self.psi + math.sqrt((m_t ** 2 *
                                                       self.phi ** 4) / 4 + v_t * self.phi ** 2 *
                                                      self.xi)) / (v_t * self.xi))
        return min(alpha_t, self.C)

    def get_params(self):
        return {'C': self.C, 'eta': self.eta, 'num_iterations': self.num_iterations}
