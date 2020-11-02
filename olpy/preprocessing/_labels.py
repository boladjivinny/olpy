import numpy as np


class LabelEncoder:
    def __init__(self, positive_label=1):
        self.y = None
        self.positive_label = positive_label
        self.labels = None

    def fit(self, y):
        self.y = y
        labels = np.unique(self.y)

        # First check that we have two values
        if len(labels) != 2:
            raise ValueError('Expected two labels. Got {} instead'.format(len(labels)))

        # Let's check now that the specified positive label is in the array
        assert self.positive_label in labels, 'The positive label ({}) has not been found in the labels'
        self.labels = (set(labels).difference({self.positive_label}).pop(), self.positive_label)

        return self

    def transform(self, return_labels=True):
        # All is okay. Now we can change
        if return_labels:
            return np.array([1 if self.y[i] == self.positive_label else -1 for i in
                             range(self.y.shape[0])]), self.labels
        else:
            return np.array([1 if self.y[i] == self.positive_label else -1 for i in range(self.y.shape[0])])

    def fit_transform(self, y, return_labels=True):
        self.fit(y)
        return self.transform(return_labels=return_labels)
