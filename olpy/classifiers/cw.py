import numpy as np
import math

from scipy.stats import norm
from . __base import OnlineLearningModel


class CW(OnlineLearningModel):
    """The Confidence-Weighted model.

    Dredze, M.; Crammer, K. & Pereira, F.
    Confidence-Weighted linear classification 
    Proc. 25th Int. Conf. Machine Learning, 
    Association for Computing Machinery, 108, 264-271
    
    Attributes:

        a (:obj:`float`, optional): Initial variance parameter, a > 0
            Defaults to 1.

        eta (:obj:`float`, optional): Mean weight value. 
            Defaults to 0.7.

        num_iterations (:obj:`int`, optional): 
            Number of iterations to run the training for. Defaults to 1.

        random_state (:obj:`int`, optional): The random seed to use 
            with the pseudo-random generator. Defaults to None.

        positive_label (:obj:`int`, optional): The number in the output
            field that represents the positive label. The value passed
            should be different than -1. Defaults to 1.

        class_weight (:obj:`dict`, optional): Represents the relative 
            weight of the labels in the data. Useful for imbalanced classification tasks.

    Raises:
        AssertionError: if positive_label is equal to -1.
    """
    
    def __init__(
        self, 
        eta=0.7, 
        a=1, 
        num_iterations=1, 
        random_state=None,
        positive_label=1, 
        class_weight=None
    ):
        super().__init__(
            num_iterations=num_iterations, 
            random_state=random_state,
            positive_label=positive_label, 
            class_weight=class_weight
        )
        self._a = a
        self._eta = eta

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
        decision = self.weights.dot(x)
        v_t = x @ np.diag(np.diag(self._sigma)) @ x.T
        m_t = y * decision
        loss = (self._phi * math.sqrt(v_t) - m_t)
        #print(loss)
        if loss > 0:
            # We scale our learning rate (alpha) using the weight/cost
            alpha_t = self.class_weight_[y] * self._get_alpha(m_t, v_t)
            u_t = 0.25 * (-alpha_t * v_t * self._phi + math.sqrt(
                alpha_t ** 2 * v_t ** 2 * self._phi ** 2 + 4 * v_t)) ** 2
            beta_t = alpha_t * self._phi / (math.sqrt(u_t) +
                                           alpha_t * self._phi * v_t)
            sigma = np.expand_dims(x @ self._sigma, axis=0)
            self.weights += alpha_t * y * np.squeeze(sigma)
            self._sigma -= beta_t * sigma.T @ sigma

    def _get_alpha(self, m_t, v_t):
        """Computes the alpha for the CW/SCW algorithms.
        
        The `alpha` variable is used to determine the magnitude of
        update that needs to be applied to the weights.

        Args:
            m_t (:obj:`float`): Represents whether there was an error in
                prediction or not. 1 for no error, -1 otherwise.
            v_t (:obj:`float`): Represents how far the point was from its
                actual value.

        Returns:
            float: the value for `alpha`.
        """
        return max(0, ((-m_t * self._psi 
                        + math.sqrt((m_t ** 2 * self._phi ** 4) 
                                    / 4 + v_t * self._phi ** 2 * self._xi)) 
                        / (v_t * self._xi)))

    def _setup(self, X: np.ndarray):
        """Initializes the values for the model' parameters.

        Based on the data in argument, this method initializes 
        the parameters `sigma`, `phi`, `psi` and `x_i`.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns

        Returns:
            None
        """
        self._sigma = self._a * np.eye(X.shape[1])
        self._phi = norm.ppf(self._eta)
        self._psi = 1 + (self._phi ** 2) / 2
        self._xi = 1 + self._phi ** 2

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

        params['a'] = self._a
        params['eta'] = self._eta

        return params
