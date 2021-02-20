import numpy as np

from . __base import OnlineLearningModel


class Perceptron(OnlineLearningModel):
    """The Perceptron model.

    Rosenblatt, F.,
    The perceptron: a probabilistic model for information storage
    and organization in the brain., Psychological review, 
    American Psychological Association, 1958, 65, 386
    
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
        prediction = -1 if self.weights.dot(x) < 0 else 1
        if y != prediction:
            self.weights = self.weights + y * x * self.class_weight_[y]
