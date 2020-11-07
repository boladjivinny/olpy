import numpy as np
from numpy import linalg as LA

from olpy._model import OnlineLearningModel


class PA(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, num_iterations=20, random_state=None, positive_label=1):
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

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        loss = max(0, 1 - y * decision)
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
        positive_label: 1 or -1
            Represents the value that is used as positive_label.

        Returns
        -------
        None
        """
        return loss / s_t if s_t > 0 else 1


class PA_I(PA):
    def __init__(self, C=1, num_iterations=20, random_state=None, positive_label=1):
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

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)
        self.C = C

    def _get_gamma(self, loss, s):
        return min(self.C, loss / s)

    def get_params(self):
        return {'C': self.C, 'num_iterations': self.num_iterations}


class PA_II(PA):
    def __init__(self, C=1, num_iterations=20, random_state=None, positive_label=1):
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

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)
        self.C = C

    def _get_gamma(self, loss, s_t):
        return loss / (s_t + (1/(2 * self.C )))

    def get_params(self):
        return {'C': self.C, 'num_iterations': self.num_iterations}