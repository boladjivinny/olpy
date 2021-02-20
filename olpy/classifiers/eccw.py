import numpy as np
import math

from scipy.stats import norm
from . cw import CW


class ECCW(CW):
    """The Exact convex confidence-weighted learning model.

    Crammer, K.; Dredze, M. & Pereira, F.
    Koller, D.; Schuurmans, D.; Bengio, Y. & Bottou, L. (Eds.)
    Exact convex confidence-weighted learning 
    Advances in Neural Information Processing Systems, 
    Curran Associates, Inc., 109, 21, 345-352
    
    Attributes:
        a (:obj:`float`, optional): Initial variance parameter, `a > 0`.
            Defaults to 1.
        eta (:obj:`float`, optional): Mean weight value. Defaults to 0.7.
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

    def __init__(self, eta=0.7, a=1, num_iterations=1, random_state=None,
                 positive_label=1, class_weight=None):
        super().__init__(
            eta=eta, 
            a=a, 
            num_iterations=num_iterations, 
            random_state=random_state,
            positive_label=positive_label, 
            class_weight=class_weight
        )

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
        return max(0, ((1 / (v_t * self._xi)) 
                        * (-m_t * self._psi 
                            + math.sqrt((m_t ** 2 * self._phi / 4 + v_t 
                            * self._phi ** 2 * self._psi)))))
