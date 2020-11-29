import numpy as np
from numpy import linalg as LA

from olpy._model import OnlineLearningModel


class ROMMA(OnlineLearningModel):
    name = "ROMMA"
    
    def __init__(self, num_iterations=20, random_state=None, positive_label=1,\
                    class_weight=None):
        """
        Instantiate a ROMMA model for training.

        Li, Y. & Long, Philip M..
        The Relaxed Online Maximum Margin Algorithm 
        Advances in neural information processing systems, 2000, 498-504

        Parameters
        ----------
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

        super().__init__(num_iterations=num_iterations, random_state=random_state, \
                            positive_label=positive_label, class_weight=class_weight)

    def _update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        if self._is_mistake(x, y):
            if LA.norm(self.weights) == 0:
                self.weights += y * x
            else:
                deno = (LA.norm(x) * LA.norm(self.weights)) ** 2 - f_t ** 2
                if deno != 0:
                    coe_1 = ((LA.norm(x) * LA.norm(self.weights)) ** 2 - y * f_t) / deno
                    coe_2 = (LA.norm(self.weights) ** 2 * (y - f_t)) / deno
                    self.weights = coe_1 * self.weights + coe_2 * x

    def _is_mistake(self, x: np.ndarray, y: int):
        """
        Evaluates whether the prediction using the current weights leads
        to a mistake or not.

        Parameters
        ----------
        x: array or np.ndarray
            Input data point
        y: int 
            Output value. Has value 1 or -1

        Returns
        -------
        bool: True if there is an error, False otherwise
        """
        return np.sign(self.weights.dot(x)) != y


class aROMMA(ROMMA):
    name = "Aggressive ROMMA"
    def _is_mistake(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        l_t = 1 - y * f_t
        return l_t > 0
