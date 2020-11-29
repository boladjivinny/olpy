import time
import random

import numpy as np

from olpy.preprocessing import LabelEncoder

class OnlineLearningModel():
    """Base class for the online learning models."""

    def __init__(self, num_iterations=20, random_state=None, positive_label=1, class_weight=None):
        """
        Initializes the values needed for all the models.

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
        self.weights = None
        self.labels = None
        self.num_iterations = num_iterations
        self.positive_label = positive_label
        self.random_state = random_state
        self.class_weight = class_weight
        self.class_weight_ = None

        # Setting the random seed

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose=True, **kwargs):
        """
        Fits the model to the (X,Y) pair passed to the function.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        Y   : array or np.ndarray
            Output variable with binary labels.
        verbose: boolean, default True
            Specifies whether the performances should be reported for 
            the different iterations.
        
        Returns
        -------
        self
        """
        positive_label = kwargs.get('positive_label', 1)

        self.weights = np.zeros(X.shape[1])
        y_transformed, self.labels = LabelEncoder(positive_label=self.positive_label)\
                                            .fit_transform(Y, return_labels=True)
        # We have the weights, with the initial encoding {0:0.3, 1:0.7}
        if self.class_weight is not None:
            self.class_weight_ = {
                -1: self.class_weight[self.labels[0]],
                self.positive_label: self.class_weight[positive_label],
            }
        # Balanced set
        else:
            self.class_weight_ = {
                -1: 1,
                1: 1
            }

        random.seed(self.random_state)
        self._setup(X)
        
        for iteration in range(1, self.num_iterations+1):
            start = time.time()
            idx = random.sample(range(X.shape[0]), k=X.shape[0])

            for x, y in zip(X[idx, :], y_transformed[idx]):
                self._update(x, y)

            if verbose:
                prediction = self.predict(X)
                print('Iteration ({}/{}) \tRuntime: {}s \tAccuracy:  {}/{}'.\
                    format(iteration, self.num_iterations, time.time() - start, \
                    np.count_nonzero(prediction==Y), X.shape[0]))
        return self

    def _update(self, x: np.ndarray, y: int):
        """
        Updates the weight vector in case a mistake occured.
        Method should be overriden by inheriting classes.

        Parameters
        ----------
        x: np.ndarray or array with size (m, 1)
            The features values for the data point.
        y: int, 1 or -1
            Output value for the data point.
        """
        return NotImplementedError

    def _setup(self, X: np.ndarray):
        """
        Performs model specific initialization that cannot be done
        in the constructor.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        Y   : array or np.ndarray
            Output variable with binary labels.
        """
        return NotImplemented

    def predict(self, X):
        """
        Predicts the label given the dataset X.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        
        Returns
        -------
        np.ndarray with dimension (n,) representing the output label
        """
        return [self.labels[0] if val <= 0 else 1 for val in X @ self.weights]

    def score(self, X, y):
        """
        Compute the score performed on the dataset.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        y   : array or np.ndarray
            Output variable with dimension (n, )
        Returns
        -------
        float: Score of the model. Default is the accuracy score.
        """
        return np.count_nonzero(self.predict(X) == y) / X.shape[0]

    def decision_function(self, X):
        """
        Compute the score performed on the dataset.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        y   : array or np.ndarray
            Output variable with dimension (n, )
        Returns
        -------
        float: Score of the model. Default is the accuracy score.
        """
        return X @ self.weights

    def get_params(self, deep=True):
        """
        Compute the score performed on the dataset.

        Parameters
        ----------
        X   : array or np.ndarray
            Input variable with dimension (n, m)
        y   : array or np.ndarray
            Output variable with dimension (n, )
        Returns
        -------
        float: Score of the model. Default is the accuracy score.
        """
        return {"num_iterations": self.num_iterations, "class_weight": self.class_weight}

    def set_params(self, **parameters):
        """
        Sets the parameters specified in the call to the function.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self