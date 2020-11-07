import numpy as np
import numpy.linalg as LA

from olpy import OnlineLearningModel


class Perceptron(OnlineLearningModel):
    def __init__(self, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiates the Perceptron model for training.

        Rosenblatt, F.,
        The perceptron: a probabilistic model for information storage and organization in the brain., 
        Psychological review, American Psychological Association, 1958, 65, 386

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
        prediction = np.sign(self.weights.dot(x))
        if y != prediction:
            self.weights = self.weights + y * x


class SecondOrderPerceptron(Perceptron):
    def __init__(self, a=1, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiate a Second Order Perceptron instance for training.

        Cesa-Bianchi, N.; Conconi, A. & Gentile, C.
        A Second-Order perceptron algorithm 
        SIAM Journal on Computing, 2005, 34, 640-668

        Parameters
        ----------
        a   : float, default 1
            Trade-off parameter. a is in the range [0,1]
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
        self.a = a

    def _setup(self, X, y):
        self.sigma = self.a * np.eye(X.shape[1])

    def _update(self, x: np.ndarray, y: int):
        sigma = x @ self.sigma
        v_t = x @ sigma.T
        beta_t = 1 / (v_t + 1)
        self.sigma = self.sigma - beta_t * sigma.T @ sigma

        prediction = np.sign(self.weights @ self.sigma @ x.T)

        if y != prediction:
            self.weights = self.weights + y * x

    def predict(self, data):
        sigma = data @ self.sigma
        v = data @ sigma.T
        beta = LA.inv(v + np.ones(data.shape[0], data.shape[1]))
        sigma_pred = self.sigma - beta @ sigma.T @ sigma

        pred = self.weights @ sigma_pred @ data.T

        return [self.labels[0] if x < 0 else 1 for x in pred]

    def get_params(self):
        return {'a': self.a, 'num_iterations': self.num_iterations}