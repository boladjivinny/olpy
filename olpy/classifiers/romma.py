import numpy as np
from numpy import linalg as LA

from . __base import OnlineLearningModel


class ROMMA(OnlineLearningModel):
    """The Relaxed Online Maximum Margin Algorithm.

    Li, Y. & Long, Philip M..
    The Relaxed Online Maximum Margin Algorithm 
    Advances in neural information processing systems, 100, 498-504
    
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
        f_t = self.weights.dot(x)
        if self._is_mistake(x, y):
            if LA.norm(self.weights) == 0:
                self.weights += y * x
            else:
                deno = ((LA.norm(x) * LA.norm(self.weights)) ** 2 
                        - f_t ** 2)
                if deno != 0:
                    coe_1 = (((LA.norm(x) * LA.norm(self.weights)) ** 2 
                                - y * f_t) / deno)
                    coe_2 = ((LA.norm(self.weights) ** 2 
                            * (y - f_t)) / deno)
                    self.weights = ((coe_1 * self.weights + coe_2 * x) 
                                    * self.class_weight_[y])

    def _is_mistake(self, x: np.ndarray, y: int):
        """
        Evaluates whether the prediction using the current weights leads
        to a mistake or not.

        Parameters
        ----------
        x: list or np.ndarray
            Input data point
        y: int 
            Output value. Has value 1 or -1

        Returns
        -------
        bool: True if there is an error, False otherwise
        """
        return np.sign(self.weights.dot(x)) != y
