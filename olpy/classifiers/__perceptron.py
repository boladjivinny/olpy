import numpy as np

from olpy import OnlineLearningModel


class Perceptron(OnlineLearningModel):
    name = "Perceptron"

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
    name = "Second Order Perceptron"
    
    def __init__(self, a=0.7, num_iterations=20, random_state=None, positive_label=1):
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

    def _setup(self, X: np.ndarray):
        self.sigma = self.a * np.identity(X.shape[1])

    def _update(self, x: np.ndarray, y: int):
        x_ = np.expand_dims(x, axis=0)
        s_t = x_ @ self.sigma.T
        v_t = x_ @ s_t.T
        beta_t = 1/(v_t + 1)
        sigma_t = self.sigma - beta_t * (s_t.T @ s_t)
        f_t = self.weights @ sigma_t @ x_.T

        hat_y_t = 1 if f_t >= 0 else 0
        if hat_y_t != y:
            self.weights = self.weights + y * x

        self.sigma = sigma_t

    def get_params(self, deep=True):
        params = super().get_params()
        params['a'] = self.a

        return params
