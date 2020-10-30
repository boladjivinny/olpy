import numpy as np

from olpy.preprocessing._labels import LabelEncoder

class Model():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self):
        return NotImplementedError

    def predict(self, input):
        return NotImplementedError

class BCModelWithLabelEncoding(Model):
    def __init__(self, X, y, positive_label=1):
        super().__init__(X, y)
        self.y, self.labels = LabelEncoder(self.y, positive_label=positive_label).fit_transform(return_labels=True)
        self.weights = np.zeros(self.X.shape[1])

    def predict(self, data):
        #with pred = data @ self.weights:
        return [self.labels[0] if val <=0 else 1 for val in data @ self.weights ]