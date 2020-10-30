import numpy as np

class LabelEncoder():
    def __init__(self, y, positive_label=1):
        self.y = y
        self.plabel = positive_label
        self.labels = np.unique(self.y)

    def fit(self):
        # First check that we have two values
        if len(self.labels) !=2:
            raise ValueError('Expected two labels. Got {} instead'.format(len(self.labels)))

        # Let's check now that the specified positive label is in the array
        assert(self.plabel in self.labels, 'The positive label ({}) has not been found in the labels')
        self.labels = (set(self.labels).difference(set(self.plabel)).pop(), self.plabel)
        
    
    def transform(self, return_labels=True):
        # All is okay. Now we can change
        if return_labels:
            return (np.array([1 if self.y[i] == self.plabel else -1 for i in range(self.y.shape[0])]), self.labels)
        else:
            return np.array([1 if self.y[i] == self.plabel else -1 for i in range(self.y.shape[0])])

    def fit_transform(self, return_labels=True):
        self.fit()
        return self.transform(return_labels=return_labels) 