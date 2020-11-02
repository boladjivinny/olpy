import numpy as np
from numpy.random import seed, permutation

from olpy.preprocessing import LabelEncoder


class Model:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y, n_iterations=20):
        return NotImplementedError

    def predict(self, X):
        return NotImplementedError


class OnlineLearningModel(Model):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.labels = None

    def fit(self, X: np.ndarray, Y: np.ndarray, positive_label=1, random_seed=32, idx=None):
        self.weights = np.zeros(X.shape[1])
        y_transformed, self.labels = LabelEncoder(positive_label=positive_label).fit_transform(Y, return_labels=True)
        seed(random_seed)
        if not idx:
            idx = permutation(X.shape[0])
        # Set up any parameter that depends on the dataset
        else:
            idx = [val - 1 for val in idx]
        self.setup(X, Y)

        for x, y in zip(X[idx, :], y_transformed[idx]):
            self.update(x, y)
        return self

    def update(self, x: np.ndarray, y: int):
        pass

    def setup(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X):
        return [self.labels[0] if val <= 0 else 1 for val in X @ self.weights]

