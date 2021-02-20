import numpy as np
from numpy import linalg as LA

from . pa import PA


class PA_I(PA):
    """Passive Aggressive-I Model.

    Crammer, K. et al., Online Passive-Aggressive algorithms, 
    Journal of Machine Learning Research, 106, 7, 551-585

    
    Attributes:
        C (:obj:`float`, optional): Aggressiveness parameter with `C>0`.
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
        C=1, 
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
        self._C = C

    def _get_gamma(self, loss, s):
        """Computes the coefficient used to update the weight vector.

        Args:
            loss(:obj:`float`): Loss incurred on the current instance.
            s_t (:obj:`float`): the L2-norm of the vector representing the
                current instance.

        Returns:
            float: the value of gamma to be used.
        """
        return min(self._C, loss / s)

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

        return params
