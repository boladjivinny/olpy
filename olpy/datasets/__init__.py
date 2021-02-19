# -*- coding: utf-8 -*-
"""Datasets management module.

This module provides an interface to load some available datasets 
bundled with the `OLPy package`_ . Its main purpose is to allow for
a quick access to the data in order to run some experiments.
Available datasets include:

* a1a
* svmguide1
* svmguide3

The original datasets are available `on this link`_

Example:

        >>> from olpy import datasets
        >>> from olpy.classifiers import IELLIP
        >>> a1a = datasets.load_a1a()
        >>> X_train, y_train = (a1a.train_data, a1a.train_target)
        >>> model = IELLIP()
        >>> model.fit(X_train, y_train)

.. _OLPy package:
   https://github.com/boladjivinny/olpy.git

.. _on this link:
   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

"""

__all__ = ['Dataset', 'load_a1a', 'load_svmguide1', 'load_svmguide3']

from . loaders import load_a1a, load_svmguide1, load_svmguide3
from . dataset import Dataset