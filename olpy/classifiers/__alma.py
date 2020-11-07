import numpy as np
import math

from numpy import linalg as LA

from olpy import OnlineLearningModel


class ALMA(OnlineLearningModel):
    def __init__(self, alpha=1.0, p=2, q=2, B=1, C=1, num_iterations=20, random_state=None, positive_label=1):
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)
        """
        Instantiate an ALMA model for training.

        Gentile, C.
        A New Approximate Maximal Margin Classification Algorithm 
        Journal of Machine Learning Research, 2001, 2, 213-242

        Parameters
        ----------
        p   : int, default 2
            ALMA's order, p > 0
        q   : int, default 2
            Dual value to p q > 0
        B,C : float,  default: 1
            Parameters of ALMA, B, C > 0
        alpha: float, default=1
            The sensitivity of the model, 0 < alpha <=1
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
        self.p = p
        self.q = q
        self.B = B
        self.C = C
        self.alpha = alpha
        
        self.k = 0

    def _update(self, x: np.ndarray, y: int):
        gamma_k = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)
        if y * self.weights.dot(x) <= (1 - self.alpha) * gamma_k:
            eta_k = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))
            self.weights = self.weights + eta_k * y * x
            norm_w = LA.norm(self.weights, ord=self.p)
            self.weights = self.weights / (max(1, norm_w))
            self.k += 1
    
    def _setup(self, X):
        self.k = 1

    def get_params(self):
        return {'p': self.p, 'q': self.q, 'B': self.B, 'C': self.C, \
                'alpha': self.alpha, 'num_iterations': self.num_iterations}