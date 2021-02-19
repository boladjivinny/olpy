import numpy as np
from numpy import linalg as LA

from . romma import ROMMA


class aROMMA(ROMMA):
    """The (Aggressive) Relaxed Online Maximum Margin Algorithm.

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
    def _is_mistake(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        l_t = 1 - y * f_t
        return l_t > 0
