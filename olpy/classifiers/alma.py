import numpy as np
import math

from numpy import linalg as LA

from . __base import OnlineLearningModel


class ALMA(OnlineLearningModel):
    """A New Approximate Maximal Margin Classification Algorithm.
    
    Gentile, C.
    A New Approximate Maximal Margin Classification Algorithm 
    Journal of Machine Learning Research, 101, 2, 213-242

    Attributes:
        p (int, optional): ALMA's order with p strictly greater than 0.
            Defaults to 2.
        C (:obj:`float`, optional): Parameter of ALMA with C strictly greater
            than 0. Defaults to 1.
        alpha (:obj:`float`, optional): The sensitivity of the model. `alpha` 
            takes values between 0 (non-inclusive). Defaults to 1.
        num_iterations (:obj:`int`, optional): Number of iterations 
            to run the training for. Defaults to 1.
        random_state (:obj:`int`, optional): The random seed to use 
            with the pseudo-random generator. Defaults to `None`.
        positive_label (:obj:`int`, optional): The number in the output
            field that represents the positive label. The value passed
            should be different than -1. Defaults to 1.
        class_weight (:obj:`dict`, optional): Represents the relative 
            weight of the labels in the data. Useful for imbalanced 
            classification tasks.

    Raises:
        AssertionError: if `positive_label` is equal to -1.

    """
    
    def __init__(
        self, 
        alpha=1.0, 
        p=2, 
        C=1, 
        num_iterations=1, 
        random_state=None,
        class_weight=None, 
        positive_label=1
    ):
        super().__init__(
            num_iterations=num_iterations, 
            random_state=random_state,
            positive_label=positive_label, 
            class_weight=class_weight
        )

        self._p = p
        self._C = C
        self._alpha = alpha

        self._B = 1
        self._k = 0

    def _update(self, x: np.ndarray, y: int):
        """Updates the weight vector in case a mistake occured.
        
        When presented with a data point, this method evaluates
        the error and based on the result, updates or not the 
        weights vector.

        Args:
            x (:obj:`np.ndarray` or `list`): An array representing
                one single data point. Array needs to be 2D.
            y (`int`): Output value for the data point. Takes value
                between 1 and -1.

        Returns:
            None

        Raises:
            IndexError: if the value x is not 2D.
        """
        gamma_k = self._B * math.sqrt(self._p - 1) / math.sqrt(self._k)
        if y * self.weights.dot(x) <= (1 - self._alpha) * gamma_k:
            eta_k = ((self._C / (math.sqrt(self._p - 1) * math.sqrt(self._k)))
                     * self.class_weight_[y])
            self.weights = self.weights + eta_k * y * x
            norm_w = LA.norm(self.weights, ord=self._p)
            self.weights = self.weights / (max(1, norm_w))
            self._k += 1

    def _setup(self, X):
        """Initializes the values for the model' parameters.

        Based on the data in argument, this method initializes 
        the parameters `k` and `B` of the ALMA algorithm.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns

        Returns:
            None
        """
        self._k = 1
        self._B = 1/self._alpha

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        This function is for use with hyper-parameter tuning utilities
        such as `GridSearchCV`_.

        Args:
            deep(:obj:`bool`, optional): If True, will return the parameters
            for this estimator and contained sub-objects that are 
            estimators. Defaults to True.

        .. _GridSearchCV:
           https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

        """
        params = super().get_params()
        params['p'] = self._p
        params['C'] = self._C
        params['alpha'] = self._alpha

        return params
