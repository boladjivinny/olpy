import numpy as np
import math

from scipy.stats import norm
from . scw import SCW


class SCW2(SCW):
    """Soft Confidence Weighted variant 2 model.

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
        n_t = v_t + 1 / (2 * self._C)
        return max(0, ((-(2 * m_t * n_t + self._phi ** 2 * m_t * v_t) +
                       math.sqrt((self._phi ** 4 * m_t ** 2 * v_t * 2 + 4 
                                  * n_t * v_t * self._phi ** 2 
                                  * (n_t + v_t * self._phi * 2)))) 
                        / (2 * (n_t ** 2 + n_t * v_t * self._phi ** 2))))

