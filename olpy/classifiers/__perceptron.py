import numpy as np

from olpy import OnlineLearningModel


class Perceptron(OnlineLearningModel):
    name = "Perceptron"

    def __init__(self, num_iterations=20, random_state=None, positive_label=1, class_weight=None):
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
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label,
                         class_weight=class_weight)

    def _update(self, x: np.ndarray, y: int):
        prediction = -1 if self.weights.dot(x) < 0 else 1
        if y != prediction:
            self.weights = self.weights + y * x * self.class_weight_[y]


class SecondOrderPerceptron(Perceptron):
    name = "Second Order Perceptron"
    
    def __init__(self, a=1, num_iterations=20, random_state=None, positive_label=1, class_weight=None):
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
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label,
                         class_weight=class_weight)
        self.a = a

    def _setup(self, X: np.ndarray):
        self.sigma = self.a * np.identity(X.shape[1])

    def _update(self, x: np.ndarray, y: int):
        x_ = np.expand_dims(x, axis=0)
        s_t = x_ @ self.sigma.T
        v_t = x @ s_t.T
        beta_t = 1/(v_t + 1)
        sigma_t = self.sigma - beta_t * (s_t.T @ s_t)
        f_t = self.weights @ sigma_t @ x_.T

        if np.sign(f_t) != y:
            self.weights = self.weights + y * x * self.class_weight_[y]
        self.sigma = sigma_t

    def get_params(self, deep=True):
        params = super().get_params()
        params['a'] = self.a

        return params
