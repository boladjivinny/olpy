import numpy as np
import numpy.linalg as LA

from olpy import OnlineLearningModel


class Perceptron(OnlineLearningModel):
    def __init__(self):
        super().__init__()

    def update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        hat_y_t = 1 if f_t >= 0 else -1
        if y != hat_y_t:
            self.weights = self.weights + y * x


class SecondOrderPerceptron(Perceptron):
    def __init__(self, a=1):
        super().__init__()
        self.a = a

    def setup(self, X, y):
        self.Sigma = self.a * np.eye(X.shape[1])

    def update(self, x: np.ndarray, y: int):
        S_x_t = x @ self.Sigma
        v_t = x @ S_x_t.T
        beta_t = 1 / (v_t + 1)
        Sigma_t = self.Sigma - beta_t * S_x_t.T @ S_x_t

        f_t = self.weights @ Sigma_t @ x.T
        hat_y_t = 1 if f_t >= 0 else -1

        if y != hat_y_t:
            self.weights = self.weights + y * x

    def predict(self, data):
        S_x = data @ self.Sigma
        v = data @ S_x.T
        beta = LA.inv(v + np.ones(data.shape[0], data.shape[1]))
        Sigma = self.Sigma - beta @ S_x.T @ S_x

        pred = self.weights @ Sigma @ data.T

        return [self.labels[0] if x < 0 else 1 for x in pred]
