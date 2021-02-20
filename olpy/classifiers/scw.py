import numpy as np
import math

from scipy.stats import norm
from . cw import CW


class SCW(CW):
    """Soft Confidence Weighted model.

    Wang, J.; Zhao, P. & Hoi, S. C. H.
        Exact Soft Confidence-Weighted learning 
        CoRR, 112, abs/1206.4612
    
    Attributes:
        eta (:obj:`float`, optional): Mean weight value. Defaults to 0.7.
        C (:obj:`float`, optional): Initial variance parameter, `C > 0`.
            Defaults to 1.
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
        eta=0.7, 
        C=1, 
        num_iterations=1, 
        random_state=None,
        positive_label=1, 
        class_weight=None
    ):
        super().__init__(
            eta=eta, 
            a=1, 
            num_iterations=num_iterations,
            random_state=random_state, 
            positive_label=positive_label,
            class_weight=class_weight
        )
        self._C = C

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
        alpha_t = max(0, ((-m_t * self._psi 
                            + math.sqrt((m_t ** 2 * self._phi ** 4) 
                                        / 4 + v_t * self._phi ** 2 * self._xi))
                            / (v_t * self._xi)))
        return min(alpha_t, self._C)

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

        params['C'] = self._C
        params['eta'] = self._eta

        # Remove parameter from parent class
        del params['a']
        return params

