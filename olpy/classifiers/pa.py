import numpy as np
from numpy import linalg as LA

from . __base import OnlineLearningModel


class PA(OnlineLearningModel):
    """Passive-Aggressive Model.

    Crammer, K. et al., Online Passive-Aggressive algorithms, 
    Journal of Machine Learning Research, 106, 7, 551-585

    
    Attributes:
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
        loss = max(0, 1 - y * decision)
        if loss > 0:
            sq_norm = LA.norm(x) ** 2
            gamma = self._get_gamma(loss, sq_norm)
            self.weights = (self.weights 
                            + gamma 
                            * y * x * self.class_weight_[y])

    def _get_gamma(self, loss, s_t):
        """Computes the coefficient used to update the weight vector.

        Args:
            loss(:obj:`float`): Loss incurred on the current instance.
            s_t (:obj:`float`): the L2-norm of the vector representing the
                current instance.

        Returns:
            float: the value of gamma to be used.
        """
        return loss / s_t if s_t > 0 else 1
