# -*- coding: utf-8 -*-
"""The repository of all the binary classifiers implemented with the package.

This module exposes a series of `online machine learning`_ models that can
be used for binary classification. The models learn by taking one data point
at a time and can achieve very good accuracy results.

One of the features of the models is to allow for usage of class weights 
during the training process. They also permit to train using a single
data point with the `partial_fit` method.


Examples:
    Training a model

    >>> from olpy.classifiers import AROW
    >>> from olpy.datasets import load_a1a
    >>> from sklearn.metrics import accuracy_score
    >>> a1a = load_a1a()
    >>> model = AROW(random_state = 32)
    >>> _ = model.fit(a1a.train_data, a1a.train_target)
    >>> prediction = model.predict(a1a.test_data)
    >>> accuracy_score(a1a.test_target, prediction)
    0.8379312572683809

    Using the weights to change the performance

    >>> model = AROW(random_state=32, class_weight=np.list([0.4, 0.6]))
    >>> _ = model.fit(a1a.train_data, a1a.train_target)
    >>> prediction = model.predict(a1a.test_data)
    >>> accuracy_score(a1a.test_target, prediction)
    0.838254296417262

    Doing a partial learn (meant for `active learning` processes)

    >>> import random
    >>> import numpy as np
    >>> random.seed(32)
    >>> model = AROW(random_state = 32)
    >>> for i in random.sample(range(a1a.train_data.shape[0]), k=10):
    ...     model = model.partial_fit(np.expand_dims(a1a.train_data[i], axis=0), [a1a.train_target[i]])
    >>> prediction = model.predict(a1a.test_data)
    >>> accuracy_score(a1a.test_target, prediction)
    0.13551492440883836


.. _online machine learning:
   https://en.wikipedia.org/wiki/Online_machine_learning

.. _ active learning:
   https://en.wikipedia.org/wiki/Active_learning_(machine_learning)

"""

__all__ = [
    'IELLIP', 'NHerd', 'OGD', 'PA', 'PA_I', 'PA_II', 'Perceptron', 
    'SecondOrderPerceptron', 'ALMA', 'AROW', 'CW', 'SCW', 'SCW2', 
    'NAROW', 'ROMMA', 'aROMMA', 'ECCW'
]

from . iellip import IELLIP
from . nherd import NHerd
from . ogd import OGD
from . pa import PA
from . pa1 import PA_I
from . pa2 import PA_II
from . perceptron import Perceptron
from . sop import SecondOrderPerceptron
from . alma import ALMA
from . arow import AROW
from . cw import CW
from . scw import SCW
from . scw2 import SCW2
from . eccw import ECCW
from . narow import NAROW
from . romma import ROMMA
from . aromma import aROMMA 
