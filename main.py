"""
This is the main module of the OLPy package.
It compares the performance of the various algorithms and returns the
result to the use in the desired format.
"""

__all__ = []
__version__ = '1.0.0'
__author__ = 'Boladji Vinny'

import argparse
import time
import os
import joblib
import pathlib

import pandas as pd
import numpy as np

from olpy.classifiers import *
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                            confusion_matrix, hinge_loss)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight


def olpy_parse_args():
    parser = argparse.ArgumentParser(
        prog="OLPy",
        description='After receiving input from the user, this program trains\
                     a series of Online Machine Learning models for binary\
                     classification.'
    )
    parser.add_argument(
        'train_set', 
        metavar='TRAINING SET',
        help='CSV file containing the training dataset.',
        type=argparse.FileType('r')
    )
    parser.add_argument(
        'test_set', 
        metavar='TESTING SET',
        help='CSV file containing the test dataset.',
        type=argparse.FileType('r')
    )
    parser.add_argument(
        '-l', 
        '--label', 
        type=str, 
        default='Label',
        help='index of the target variable. (default:  %(default)s)'
    )
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+', 
        default='all',
        help='the list of models to try from or use use %(default)s',
        choices=[
            'all', 'alma', 'arow', 'cw', 'scw', 'scw2', 'iellip', 'narow',
            'nherd', 'ogd', 'pa', 'pa1', 'pa2', 'perceptron', 'sop', 'romma',
            'aromma'
        ]
    )
    parser.add_argument(
        '-n', 
        type=int, 
        default=1, 
        help='the number of iterations to run. (default: %(default)s)'
    )
    parser.add_argument(
        '-s', 
        type=int, 
        default=None, 
        help='the random seed to use in training the models. \
            (default: %(default)s)'
    )
    parser.add_argument(
        '-o', 
        type=str, 
        default='experiment-results.csv',
        help='file to which the reports would be saved \
            (default: %(default)s)'
    )
    parser.add_argument(
        '-b', 
        '--bias', 
        help="whether or not a bias should be used for the training.", 
        action="store_true"
    )
    parser.add_argument(
        '-w', 
        '--use-weights', 
        help="whether or not  weights should be used while training the\
             models.",
        action="store_true"
    )
    parser.add_argument(
        '--weights', 
        help="custom weights to use with the training", 
        type=float,
        nargs='+'
    )
    parser.add_argument(
        '--cv', 
        help="whether or not hyper-parameter through cross validation should\
             be done.", 
        action="store_true"
    )
    parser.add_argument(
        '-d',
        '--dump-dir',
        type=pathlib.Path, 
        default='.',
        help="output directory for dumping the models. (default: %(default)s)"
    )
    parser.add_argument(
        '-v',
        help='represents the verbosity level of the application. \
            (default: %(default)d)', 
        action="count",
        default=0
    )
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s {}'.format(__version__) 
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = olpy_parse_args()

    # Collect the arguments
    train_file = args.train_set
    test_file = args.test_set
    verbose = args.v
    output_file = args.o
    seed = args.s
    models = args.models
    n_iterations = args.n
    label = args.label
    bias = args.bias
    use_weights = args.use_weights
    cv = args.cv
    model_dir = args.dump_dir
    weights = args.weights

    # Load the datasets
    scaler = MinMaxScaler()
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    if bias:
        train_data.insert(0, 'Bias', np.ones(train_data.shape[0]))
        test_data.insert(0, 'Bias', np.ones(test_data.shape[0]))

    # Scaling the dataset to avoid numerical issues
    Y_train = train_data.loc[:, label].to_numpy()
    X_train = scaler.fit_transform(train_data.drop(columns=[label]))
    Y_test = test_data.loc[:, label].to_numpy()
    X_test = scaler.fit_transform(test_data.drop(columns=[label]))

    # Check the oversampling now
    class_weight = None
    if use_weights:
        if weights is not None and len(weights) >= 2:
            class_weight = np.array(weights)
        else:
            class_weight = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(Y_train), 
                y=Y_train
            )
        

    # First we replace all by the list of available models
    if models == 'all' or 'all' in models:
        models = [
            'alma', 'arow', 'cw', 'scw', 'scw2', 'iellip', 'narow', 'nherd',
            'ogd', 'pa', 'pa1', 'pa2', 'perceptron', 'sop', 'romma', 'aromma'
        ]

    # Create a variable to store the model objects
    models_ = []
    params_ = []

    for model in models:
        model = model.lower()
        if model == 'alma':
            models_.append(ALMA(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'p': range(2, 12, 2),
                'alpha': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'arow':
            models_.append(AROW(random_state=seed))
            params_.append({
                'r': [2 ** i for i in range(-4, 5)]
            })
        if model == 'cw':
            models_.append(CW(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1, 0.1)),
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw':
            models_.append(SCW(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw2':
            models_.append(SCW2(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'iellip':
            models_.append(IELLIP(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'b': list(np.arange(0.1, 1.1, 0.1)),
                'c': list(np.arange(0.1, 1.0, 0.1))
            })
        if model == 'narow':
            models_.append(NAROW(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
            })
        if model == 'nherd':
            models_.append(NHerd(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'ogd':
            models_.append(OGD(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa':
            models_.append(PA(random_state=seed))
            params_.append({})
        if model == 'pa1':
            models_.append(PA_I(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa2':
            models_.append(PA_II(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'perceptron':
            models_.append(Perceptron(random_state=seed))
            params_.append({})
        if model == 'sop':
            models_.append(SecondOrderPerceptron(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1))
            })
        if model == 'romma':
            models_.append(ROMMA(random_state=seed))
            params_.append({})
        if model == 'aromma':
            models_.append(aROMMA(random_state=seed))
            params_.append({})

    summary = pd.DataFrame(
        np.zeros((len(models_), 10)), 
        columns=[
            'Training-Time', 'Prediction-Time', 'Accuracy', 'F1-Score', 
            'Recall', 'ROC_AUC-Score', 'FP', 'FN', 'TP', 'TN'
        ])
    summary.insert(0, 'Model', [model for model in models])

    if verbose > 0:
        print(
            "%9s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\n" %
            (
                  'algorithm', 'train time (s)', 'test time (s)', 'accuracy',
                  'f1-score', 'roc-auc','true positive', 'true negative', 
                  'false positive', 'false negative'
            ))

    i = 0
    best_params_record = "Best params: \n"
    for model in models_:
        if use_weights:
            model.set_params(class_weight=class_weight)
        # Use the verbose level from the command line
        if cv:
            model_ = GridSearchCV(model, params_[i], n_jobs=-1)
            model_.fit(X_train, Y_train, verbose=verbose-1)
            # After collecting, let's save, report and proceed
            model.set_params(**model_.best_params_)
            best_params_record += (model.name + "\n" 
                                   + str(model_.best_params_) + "\n\n")

        # Set the number of iterations now
        model.set_params(num_iterations=n_iterations)
        training_start = time.time()

        model.fit(X_train, Y_train, verbose=False)
        duration = time.time() - training_start

        scores = model.decision_function(X_test)
        test_start = time.time()
        preds = model.predict(X_test)
        preds_duration = time.time() - test_start

        acc = accuracy_score(Y_test, preds)
        f1 = f1_score(Y_test, preds)
        tn, fp, fn, tp = confusion_matrix(Y_test, preds,
                             normalize='true').ravel()

        # ROC would not compute if for instance we have only one class in the
        # test data.
        # This is the case for the svmguide3 dataset bundled with the package.
        try:
            roc = roc_auc_score(Y_test, scores)
        except ValueError:
            roc = np.nan

        if verbose:
            print(
                "%-12s\t%-3f\t%-3f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f" 
                %
                (list(
                    set(models))[i], duration, preds_duration,
                     acc, f1, roc, tp, tn, fp, fn
                ))

        summary.loc[i, 'Training-Time'] = duration
        summary.loc[i, 'Prediction-Time'] = preds_duration
        summary.loc[i, 'Accuracy'] = acc
        summary.loc[i, 'F1-Score'] = f1
        summary.loc[i, 'ROC_AUC-Score'] = roc
        summary.loc[i, 'TP'] = tp
        summary.loc[i, 'TN'] = tn
        summary.loc[i, 'FP'] = fp
        summary.loc[i, 'FN'] = fn

        if model_dir is not None:
            # Save the model
            dump_name = list(set(models))[i] + '.dump'
            joblib.dump(model, model_dir  / dump_name)

        i = i + 1

    if cv:
        print()
        print()
        print(best_params_record)

    if output_file is not None:
        summary.to_csv(output_file, index=False)
