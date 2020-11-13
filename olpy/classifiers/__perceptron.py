import numpy as np
import numpy.linalg as LA

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
        #self.sigma = self.a * np.eye(X.shape[1])
        self.S = []

    def _update(self, x: np.ndarray, y: int):
        #print(self.S.shape, x.shape)
        S_t = self.S
        S_t.append(x)
        S_t_ = np.array(S_t)
        S_t_ = S_t_.T
        w = LA.inv(self.a * np.identity(len(x)) + S_t_ @ S_t_.T)
        w = w @ self.weights
        prediction = np.sign(w.T @ x)
        if prediction == 0:
            prediction = 1
        if y != prediction:
            self.weights = self.weights + y * x
            self.S = S_t

    def predict(self, data):
        S_t = (np.array(self.S)).T
        w = LA.inv(self.a * np.identity(data.shape[1]) + S_t @ S_t.T)
        w = w @ self.weights
        prediction = np.sign(w.T @ data.T)
        
        return [0 if pred < 0 else 1 for pred in prediction]

    def get_params(self):
        return {'a': self.a, 'num_iterations': self.num_iterations}