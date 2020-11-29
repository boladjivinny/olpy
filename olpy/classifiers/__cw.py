import numpy as np
import math

from scipy.stats import norm
from olpy._model import OnlineLearningModel


class CW(OnlineLearningModel):
    name = "Confidence-Weighted"

    def __init__(self, eta=0.7, a=1, num_iterations=20, random_state=None,
                 positive_label=1, class_weight=None):
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
            Mean weight value.
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.
        class_weight: dict
            Represents the relative weight of the labels in the dataset.
            Useful for imbalanced classification tasks.

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)
        self.a = a
        self.eta = eta

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        v_t = x @ np.diag(np.diag(self.sigma)) @ x.T
        m_t = y * decision
        loss = (self.phi * math.sqrt(v_t) - m_t) * self.class_weight_[y]
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

    def get_params(self, deep=True):
        params = super().get_params()

        params['a'] = self.a
        params['eta'] = self.eta

        return params


class SCW(CW):
    name = "Soft Confidence-Weighted"

    def __init__(self, eta=0.7, C=1, num_iterations=20, random_state=None,
                 positive_label=1, class_weight=None):
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
            Mean weight value.
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.
        class_weight: dict
            Represents the relative weight of the labels in the dataset.
            Useful for imbalanced classification tasks.

        Returns
        -------
        None
        """
        super().__init__(eta=eta, a=1, num_iterations=num_iterations,
                         random_state=random_state, positive_label=positive_label,
                         class_weight=class_weight)
        self.C = C

    def _get_alpha(self, m_t, v_t):
        alpha_t = max(0, (-m_t * self.psi + math.sqrt((m_t ** 2 *
                                                       self.phi ** 4) / 4 + v_t * self.phi ** 2 *
                                                      self.xi)) / (v_t * self.xi))
        return min(alpha_t, self.C)

    def get_params(self, deep=True):
        params = super().get_params()

        params['C'] = self.C
        params['eta'] = self.eta

        del params['a']
        return params


class SCW2(SCW):
    name = "Soft Confidence Weighted (version 2)"

    def _get_alpha(self, m_t, v_t):
        n_t = v_t + 1 / (2 * self.C)
        return max(0, (-(2 * m_t * n_t + self.phi ** 2 * m_t * v_t) +
                       math.sqrt(self.phi ** 4 * m_t ** 2 * v_t * 2 + 4 *
                                 n_t * v_t * self.phi ** 2 * (n_t + v_t * self.phi * 2))) /
                   (2 * (n_t ** 2 + n_t * v_t * self.phi ** 2)))


class ECCW(CW):
    name = "Exact Convex Confidence-Weighted Learning"

    def __init__(self, eta=0.7, a=1, num_iterations=20, random_state=None,
                 positive_label=1, class_weight=None):
        """
        Instantiate a Confidence Weighted model for training.
        
        Crammer, K.; Dredze, M. & Pereira, F.
        Koller, D.; Schuurmans, D.; Bengio, Y. & Bottou, L. (Eds.)
        Exact convex confidence-weighted learning 
        Advances in Neural Information Processing Systems, 
        Curran Associates, Inc., 2009, 21, 345-352

        Parameters
        ----------
        a   : float, default 1
            Initial variance parameter, a > 0
        eta : float, default=0.7
            Mean weight value. 0.5 <= mu <= 1
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.
        class_weight: dict
            Represents the relative weight of the labels in the dataset.
            Useful for imbalanced classification tasks.

        Returns
        -------
        None
        """
        super().__init__(eta=eta, a=a, num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)

    def _get_alpha(self, m_t, v_t):
        return max(0, (1 / (v_t * self.xi)) * (-m_t * self.psi +
                                               math.sqrt(m_t ** 2 * self.phi / 4 + v_t * self.phi ** 2 * self.psi)))
