import time
import random

import numpy as np

from olpy.preprocessing import LabelEncoder
from olpy.exceptions import NotFittedError


class OnlineLearningModel:
    """Base class for the online learning models.
    
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
        self.weights = None
        self.labels = None
        self.num_iterations = num_iterations
        # Safe check to avoid having the same label 
        # as positive and negative
        assert positive_label != -1
        self.positive_label = positive_label
        self.random_state = random_state
        self.class_weight = class_weight
        self.class_weight_ = {
            -1: 1,
            1: 1
        }

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose=False, **kwargs):
        """Fits the model to the (X,Y) pair passed to the function.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns
            Y (:obj:`numpy.ndarray`): Output variable with binary
                labels.
            verbose (:obj:`bool`, optional): Specifies whether the 
                performances should be reported or not. Defaults to False.
            ** kwargs: Arbitrary keyword arguments.
        
        Returns:
            self: the trained model.
        """
        positive_label = kwargs.get('positive_label', 1)

        self.weights = np.zeros(X.shape[1])
        y_transformed, self.labels = LabelEncoder(
            positive_label=self.positive_label).fit_transform(
            Y,
            return_labels=True
        )
        if self.class_weight is not None:
            weights = self.class_weight / sum(self.class_weight)
            self.class_weight_ = {
                -1: weights[self.labels[0]],
                1: weights[positive_label],
            }
        random.seed(self.random_state)
        self._setup(X)

        for iteration in range(1, self.num_iterations + 1):
            start = time.time()
            idx = random.sample(range(X.shape[0]), k=X.shape[0])
            for x, y in zip(X[idx, :], y_transformed[idx]):
                self._update(x, y)

            if verbose:
                prediction = self.predict(X)
                print('Iteration ({}/{}) \tRuntime: {}s \tAccuracy:  {}/{}'.
                    format(
                    iteration, self.num_iterations,
                    time.time() - start,
                    np.count_nonzero(prediction == Y), X.shape[0]))
        return self

    def partial_fit(self, X, Y, classes=None):
        """Trains the model on a single data point.

        Args:

            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns
            Y (:obj:`numpy.ndarray`): Output variable with binary
                labels.

            classes (:obj:`list` or `tuple`, optional): Represents the
                available labels in the dataset. Needs to be passed only
                once.
        
        Returns:
            self: the trained model.
        """
        self.labels = classes if classes else [0, 1]

        if self.weights is None:
            self._setup(X)
            self.weights = np.zeros(X.shape[1])

        for x, y in zip(X, Y):
            # Set the value to -1 
            if y != self.positive_label:
                y = -1
            self._update(x.squeeze(), y)
        return self

    def _update(self, x: np.ndarray, y: int):
        """Update the weight vector in case a mistake occured.
        
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
        return NotImplementedError

    def _setup(self, X: np.ndarray):
        """Perform model's specific initialization steps.

        This methods performs a series of initialization that cannot
        be done properly in the constructor. It is implemented 
        differently depending on the model.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns

        Returns:
            None
        """
        return NotImplemented

    def predict(self, X):
        """Predict the label given the test dataset X.

        Given an unlabelled data, this function predicts the labels to
        be assigned to it based on the weights learned so far.

        Args:
            X (:obj:`numpy.ndarray` or `list`): unlabelled data.
        
        Returns:
            np.ndarray with dimension (n,) representing the output 
            labels.

        Raises:
            NotFittedError: when the method is called without prior
            fitting.
        """
        if self.weights is None : 
            raise NotFittedError("model instance of {self.__class__.__name__}\
                                 is untrained",
                                 "Attempted to predict using the model", 
                                 "This model has not yet been fitted")
        return [self.labels[0] if val <= 0 else 1 for val in X @ self.weights]

    def score(self, X, y):
        """Compute the score performed on the dataset.

        Given a labelled data, this method evaluate the performance of
        the models by predicting and computing the accuracy score.
        
        Note:
            Different models can override the function to return a 
            different metric.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns
            Y (:obj:`, `numpy.ndarray`): Output variable with binary
                labels.

        Returns:
            float: The accuracy score of the model given the labelled
            data.
        """
        return np.count_nonzero(self.predict(X) == y) / X.shape[0]

    def decision_function(self, X):
        """Compute the values of the decision boundaries.

        Given an unlabelled data, this method computes the probabilities
        of being assigned the positive label.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                    m columns

        Returns:
            list[float]: The probabilities for each data point to be
            of the positive class label.
        """
        return X @ self.weights

    def predict_proba(self, X):
        """Compute the probability that a model is from a class.

        Given an unlabelled data, this method returns the probability
        for each data point to belong to either class.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                    m columns

        Returns:
            :obj:`list` of size (n, 2): the probabilities of the 
                data points belonging to each class.
        """
        pred = (X @ self.weights).tolist()
        probs = []
        probs.append([1 - p for p in pred])
        probs.append(pred)
        return probs

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
        return {
            "num_iterations": self.num_iterations,
            "class_weight": self.class_weight
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested 
        objects (such as Pipeline). The latter have parameters of the
        form `<component>__<parameter>` so that it’s possible to update
        each component of a nested object.

        Args:
            ** parameters (:obj:`dict`): Estimator parameters.

        Returns:
            self: estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
