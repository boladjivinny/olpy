import numpy as np

from . __base import OnlineLearningModel


class NHerd(OnlineLearningModel):
    """Gaussian Herding model.

    Crammer, K. & Lee, D., Learning via gaussian herding, Advances
    in Neural Information Processing Systems, Curran Associates, 
    Inc., 110, 23, 451-459
    
    Attributes:
        a (:obj:`float`, optional): Trade-off parameter. `a` is in the range
            `[0,1]`. Defaults to 1.
        C (:obj:`float`, optional): Gaussia Herding's parameter with `C > 0`.
            Defaults to 1.
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
        a=1, 
        C=1, 
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
        self._a = a
        self._C = C
        self._sigma = None

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
        v = x @ self._sigma @ x.T
        m = y * decision
        loss = (1 - m)
        if loss > 0:
            beta = 1 / (v + 1/self._C)
            # Scale using the class weights. The higher the weight, higher the
            # alpha.
            alpha = max(0, 1 - m) * beta * self.class_weight_[y]
            sigma = np.expand_dims(x @ self._sigma.T, axis=0)
            self.weights = self.weights + alpha * y * np.squeeze(sigma)
            self._sigma = (self._sigma 
                          - ((beta ** 2) 
                            * (v + 2 * (1/self._C)) 
                            * sigma.T @ sigma))

    def _setup(self, X: np.ndarray):
        """Initializes the values for the model' parameters.

        Based on the data in argument, this method initializes 
        the covariance matrix `sigma`.

        Args:
            X (:obj:`numpy.ndarray`): Input data with n rows and
                m columns

        Returns:
            None
        """
        self._sigma = self._a * np.eye(X.shape[1])

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
        params['a'] = self._a
        params['C'] = self._C

        return params
