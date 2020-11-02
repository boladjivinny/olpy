import numpy as np

from numpy.random import permutation
from olpy.classifiers import *


class OneShotLearning:
    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs.get('p', None)
        self.C = kwargs.get('C', None)
        self.b = kwargs.get('b', None)
        self.eta = kwargs.get('eta', None)

    def fit(self, X: np.ndarray, y: np.ndarray):
        ids = permutation(X.shape[0])


def find_parameters(object):
    if isinstance(object, AROW):
        print("Got arrow")

def find_best_parameter_eta():
    return 1