import numpy as np
import math

from sklearn.metrics import zero_one_loss, log_loss, mean_squared_error, hinge_loss
from olpy import OnlineLearningModel


class OGD(OnlineLearningModel):
    name = "Online Gradient Descent"
    
    def __init__(self, C=1, loss_function=zero_one_loss, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiate an Online Gradient Descent model for training.

        This function creates an instance of the OGD online learning
        algorithm.
        
        Zinkevich, M., Online convex programming and generalized 
        infinitesimal gradient ascent, Proc. 20th Int. Conf. 
        Machine Learning, 2003, 928-936

        Parameters
        ----------
        C   : float, default 1
            Online Gradient Descent's parameter
        loss_function : callable, default sklearn.metrics.zero_one_loss
            Loss function used to evaluate the need to update the model
            or not
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
        self.loss_function = loss_function
        self.t = 0

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        prediction = np.sign(decision)
        C = self.C / math.sqrt(self.t)

        # Changed the parameters to call the loss function as it seems they
        # expect at least two values
        if self.loss_function == hinge_loss:
            loss = self.loss_function([y, -y], [decision, -decision])
        else:
            loss = self.loss_function([y], [prediction])

        if loss > 0:
            if self.loss_function == log_loss:
                self.weights = self.weights + C * y * x * (1 / (1 + math.exp(y * decision)))
            elif self.loss_function == mean_squared_error:
                self.weights = self.weights - C * (decision - y) * x
            else:
                self.weights = self.weights + C * y * x

        self.t += 1

    def get_params(self, deep=True):
        return {'C': self.C, 'loss_function': self.loss_function,\
            'num_iterations': self.num_iterations, 'random_state': \
                self.random_state}

    def _setup(self, X):
        self.t = 1