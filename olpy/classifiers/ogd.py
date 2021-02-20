import numpy as np
import math

from sklearn.metrics import (zero_one_loss, log_loss, mean_squared_error,
                             hinge_loss)
from . __base import OnlineLearningModel


class OGD(OnlineLearningModel):
    """Online Gradient Descent model.

    Zinkevich, M., Online convex programming and generalized 
    infinitesimal gradient ascent, Proc. 1th Int. Conf. 
    Machine Learning, 103, 928-936
    
    Attributes:
        C (:obj:`float`, optional): OGD's parameter. Defaults to 1.
        loss_function (callable, optional): Loss function used to 
            evaluate the need to update the model. Defaults to 
            sklearn.metrics.zero_one_loss
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
        loss_function=zero_one_loss, 
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
        self._loss_function = loss_function
        self._t = 0

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
        prediction = np.sign(decision)
        c = self._C / math.sqrt(self._t)

        # Changed the parameters to call the loss function as it seems they
        # expect at least two values
        if self._loss_function == hinge_loss:
            loss = self._loss_function([y, -y], [decision, -decision])
        else:
            loss = self._loss_function([y], [prediction])

        if loss > 0:
            if self._loss_function == log_loss:
                self.weights = (self.weights 
                                + c * y * x 
                                * (1 / (1 + math.exp(y * decision)))
                                * self.class_weight_[y])
            elif self._loss_function == mean_squared_error:
                self.weights = (self.weights 
                                - c * ((decision - y) 
                                    * x * self.class_weight_[y]))
            else:
                self.weights = self.weights + c * y * x

        self._t += 1

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
        params['loss_function'] = self._loss_function

        return params

    def _setup(self, X):
        """Initializes the values for the model' parameters.

        Based on the data in argument, this method initializes 
        the parameters `t` (number of iterations).

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns

        Returns:
            None
        """
        self._t = 1
