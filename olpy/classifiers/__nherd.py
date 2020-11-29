import numpy as np

from olpy import OnlineLearningModel


class NHerd(OnlineLearningModel):
    name = "Normal Herd"
    
    def __init__(self, a=1, C=1, num_iterations=20, random_state=None, \
                    positive_label=1, class_weight=None):
        """
        Instantiate an Normal Herd model for training.

        This function creates an instance of the NHerd online learning
        algorithm.

        Crammer, K. & Lee, D., Learning via gaussian herding, Advances
        in Neural Information Processing Systems, Curran Associates, 
        Inc., 2010, 23, 451-459

        Parameters
        ----------
        a   : float, default 1
            Trade-off parameter. a is in the range [0,1]
        C : float, default 1
            Normal Herd's parameter. C > 0
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
        self.a = a
        self.C = C
        self.sigma = None

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        v = x @ self.sigma @ x.T
        m = y * decision
        loss = (1 - m) * self.class_weight_[y]
        if loss > 0:
            beta = 1 / (v + 1/self.C)
            alpha = max(0, 1 - m) * beta
            sigma = np.expand_dims(x @ self.sigma.T, axis=0)
            self.weights = self.weights + alpha * y * np.squeeze(sigma)
            self.sigma = self.sigma - (beta ** 2) * (v + 2 * (1/self.C)) * sigma.T @ sigma

    def _setup(self, X: np.ndarray):
        self.sigma = self.a * np.eye(X.shape[1])

    def get_params(self, deep=True):
        params = super().get_params()
        params['a'] = self.a
        params['C'] = self.C

        return params