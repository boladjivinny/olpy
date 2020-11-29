import numpy as np
from numpy import linalg as LA

from olpy._model import OnlineLearningModel


class PA(OnlineLearningModel):
    name = "Passive-Aggressive"

    def __init__(self, num_iterations=20, random_state=None, positive_label=1,
                 class_weight=None):
        """
        Instantiates the Passive-Aggressive Model for training.

        This function creates an instance of the Passive
        Aggressive online learning algorithm.

        Crammer, K. et al., Online Passive-Aggressive algorithms, 
        Journal of Machine Learning Research, 2006, 7, 551-585

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
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        loss = max(0, 1 - y * decision) * self.class_weight_[y]
        if loss > 0:
            sq_norm = LA.norm(x) ** 2
            gamma = self._get_gamma(loss, sq_norm)
            self.weights = self.weights + gamma * y * x

    def _get_gamma(self, loss, s_t):
        """
        Computes the value of the coefficient used to update the
        weight vector.

        Parameters
        ----------
        loss: float
            Loss incurred on the current instance.
        s_t:   int, default None
            Seed for the pseudorandom generator

        Returns
        -------
        None
        """
        return loss / s_t if s_t > 0 else 1


class PA_I(PA):
    name = "Passive-Aggressive-I"
    
    def __init__(self, C=1, num_iterations=20, random_state=None, positive_label=1,
                 class_weight=None):
        """
        Instantiates the Passive-Aggressive-I Model for training.

        This function creates an instance of the Passive-Aggressive-I
        online learning algorithm.

        Crammer, K. et al., Online Passive-Aggressive algorithms, 
        Journal of Machine Learning Research, 2006, 7, 551-585

        Parameters
        ----------
        C: float, C > 0
            Aggressiveness parameter
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
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)
        self.C = C

    def _get_gamma(self, loss, s):
        return min(self.C, loss / s)

    def get_params(self, deep=True):
        params = super().get_params()
        params['C'] = self.C

        return params


class PA_II(PA):
    name = "Passive-Aggressive-II"
    
    def __init__(self, C=1, num_iterations=20, random_state=None, positive_label=1,
                 class_weight=None):
        """
        Instantiates the Passive-Aggressive-II Model for training.

        This function creates an instance of the Passive-Aggressive-II
        online learning algorithm.

        Crammer, K. et al., Online Passive-Aggressive algorithms, 
        Journal of Machine Learning Research, 2006, 7, 551-585

        Parameters
        ----------
        C: float, C > 0
            Aggressiveness parameter
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
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)
        self.C = C

    def _get_gamma(self, loss, s_t):
        return loss / (s_t + (1/(2 * self.C)))

    def get_params(self, deep=True):
        params = super().get_params()
        params['C'] = self.C

        return params
