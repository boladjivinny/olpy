import argparse
import time
import os
import sys
import joblib
import pathlib

import pandas as pd
import numpy as np

from olpy.classifiers import *
from olpy.datasets import load_a1a, load_svmguide1, load_svmguide3
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, hinge_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from querier import CommiteeQuerier

if __name__=='__main__':
    # First thing load the dataset
    a1a = load_svmguide3()

    label = 'Label'
    scaler = MinMaxScaler()
    Y_test = a1a.test_target
    X_test = a1a.test_data
    Y_train = a1a.train_target
    X_train = a1a.train_data

    models = [
        HoeffdingAdaptiveTreeClassifier(seed=32),
        SCW(random_state=32),
        ALMA(random_state=32),
        # IELLIP(random_state=32),
        # aROMMA(random_state=32),
        # NHerd(random_state=32),
        # OGD(random_state=32),
        # PA(random_state=32),
        # PA_I(random_state=32),
        # PA_II(random_state=32),
        # Perceptron(random_state=32),
        # SecondOrderPerceptron(random_state=32),
        # CW(random_state=32),
        # AROW(random_state=32),
        # SCW(random_state=32),
        # SCW2(random_state=32),
        # NAROW(random_state=32),
        # ROMMA(random_state=32),
    ]

    names = [
        'tree_classifier',
        'scw',
        'alma',
        # 'iellip',
        # 'aromma',
        # 'nherd',
        # 'ogd',
        # 'pa',
        # 'pa1',
        # 'pa2',
        # 'perceptron',
        # 'sop',
        # 'cw',
        # 'arow',
        # 'scw',
        # 'scw2',
        # 'narow',
        # 'romma',
    ]

    summary = {name: [] for name in names}
    q = CommiteeQuerier(X_train, Y_train)
    
    for _ in range(X_train.shape[0]):
        # Get the next element
        q.fetch(models)
        x, y = next(q)
        i = 0
        for i in range(len(models)):
            try:
                models[i] = models[i].partial_fit(x, y, classes=[1, -1])
            except AttributeError:
                # For river
                models[i] = models[i].learn_one({i: x[i] for i in range(len(x))}, y)

            # Predict on the test dataset and then report        
            _preds = None
            try:
                _preds = models[i].predict(X_test)
            except AttributeError:
                _preds = [0 if not models[i].predict_one(x) else models[i].predict_one(x) for x in X_test]
            finally:
                acc = accuracy_score(Y_test, _preds)
                tn, fp, fn, tp = confusion_matrix(Y_test, _preds, normalize='true', labels=np.unique(a1a.train_target)).ravel()
                #print(f"{names[i]} - {_} \t {acc} \t {tp} \t {fn} \t {tn} \t {fp}")
                summary[names[i]].append([acc, tp, fn, tn, fp])

    i = 0
    for name in names:
        df = pd.DataFrame(summary[name], columns = ['Accuracy', 'TP', 'FN', 'TN', 'FP'])
        df.to_csv(f'/tmp/{name}.csv', index=True)
        joblib.dump(models[i], f'/tmp/{name}.dump')
        i += 1
