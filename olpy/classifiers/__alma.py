import numpy as np
import math

from numpy import linalg as LA

from olpy import OnlineLearningModel


class ALMA(OnlineLearningModel):
    name = "ALMA"

    def __init__(self, alpha=1.0, p=2, C=1, num_iterations=20, random_state=None,
                 class_weight=None, positive_label=1):
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)
        """
        Instantiate an ALMA model for training.

        Gentile, C.
        A New Approximate Maximal Margin Classification Algorithm 
        Journal of Machine Learning Research, 2001, 2, 213-242

        Parameters
        ----------
        p   : int, default 2
            ALMA's order, p > 0
        C : float,  default: 1
            Parameter of ALMA, C > 0
        alpha: float, default=1
            The sensitivity of the model, 0 < alpha <=1
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
        self.p = p
        self.C = C
        self.alpha = alpha

        self.B = 1
        self.k = 0

    def _update(self, x: np.ndarray, y: int):
        gamma_k = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)
        if y * self.weights.dot(x) <= (1 - self.alpha) * gamma_k:
            eta_k = (self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))) * self.class_weight_[y]
            self.weights = self.weights + eta_k * y * x
            norm_w = LA.norm(self.weights, ord=self.p)
            self.weights = self.weights / (max(1, norm_w))
            self.k += 1

    def _setup(self, X):
        self.k = 1
        self.B = 1/self.alpha

    def get_params(self, deep=True):
        params = super().get_params()
        params['p'] = self.p
        params['C'] = self.C
        params['alpha'] = self.alpha

        return params
