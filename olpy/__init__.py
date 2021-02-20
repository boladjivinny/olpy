"""This is the main module of the OLPy package.
It compares the performance of the various algorithms and returns the
result to the use in the desired format.
"""

__all__ = ['olpy_parse_args', 'run_experiments', 'classifiers', 'datasets', 'exceptions', 'preprocessing', 'utils']
__version__ = '1.0.0'
__author__ = 'Boladji Vinny'


from . import classifiers, datasets, preprocessing, utils, exceptions

import argparse
import time
import os
import joblib
import pathlib
import sklearn

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV as __GridSearchCV



def olpy_parse_args():
    """Parses the command-line arguments passed to the main program.

    Args:
        None

    Returns:
        :obj:`Namespace`, the set of arguments parsed.

    Raises:
        KeyError: if one argument key is not existent.
    """
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


def run_experiments(
    train_file,
    test_file,
    models,
    n_iterations=1,
    label = 'Label',
    bias = False,
    use_weights = False,
    weights = None,
    cv = False,
    model_dir = None,
    verbose = False,
    output_file = None,
    seed = None,
):
    """Run an experiment using the data passed by the user.

    Given the parameters passed by the user, this function executes an
    experiment and reports the results to the user at the specified
    destination(files and/or console).

    Args:
        train_file (:obj:`str`): path to the training dataset file.
        test_file (:obj:`str`): path to the testing dataset file.
        models (:obj:`list`): a list of models to try out.
        n_iterations (:obj:`int`, optional): number of iterations to 
            run each model for. Defaults to 1.
        label (:obj:`str`, optional): the column index of the output
            variable. Defaults to 'Label'.
        bias (:obj:`bool`, optional): whether a bias should be used or
            not. Defaults to False.
        use_weights (:obj:`bool`, optional): whether weights should be
            used while training the models. Defaults to False.
        weights (:obj:`numpy.ndarray`, optional): an array representing
            the weights to use during the training process. This only 
            works when `use_weights` is set to True. 
        cv (:obj:`bool`, optional): whether cross validation will be
            ran or not. Defaults to True.
        model_dir (:obj:`str`, optional): the directory to which the 
            dumps of the models will be saved. Defaults to None.
        verbose (:obj:`bool`, optional): whether the program should produce
            output or not. Defaults to False.
        output_file (:obj:`str`): path to the output file if any.
        seed (:obj:`int`, optional): random-generator seed. Defaults 
            to None.

    Returns:
        None

    Raises:
        FileNotFoundError: if one of the files passed does not exist
        IndexError: if the specified label does not exist in the data.
        
        
    """
    # Load the datasets
    scaler = sklearn.preprocessing.MinMaxScaler()
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
            class_weight = sklearn.utils.class_weight.compute_class_weight(
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
            models_.append(classifiers.ALMA(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'p': range(2, 12, 2),
                'alpha': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'arow':
            models_.append(classifiers.AROW(random_state=seed))
            params_.append({
                'r': [2 ** i for i in range(-4, 5)]
            })
        if model == 'cw':
            models_.append(classifiers.CW(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1, 0.1)),
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw':
            models_.append(classifiers.SCW(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw2':
            models_.append(classifiers.SCW2(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'iellip':
            models_.append(classifiers.IELLIP(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'b': list(np.arange(0.1, 1.1, 0.1)),
                'c': list(np.arange(0.1, 1.0, 0.1))
            })
        if model == 'narow':
            models_.append(classifiers.NAROW(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
            })
        if model == 'nherd':
            models_.append(classifiers.NHerd(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'ogd':
            models_.append(classifiers.OGD(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa':
            models_.append(classifiers.PA(random_state=seed))
            params_.append({})
        if model == 'pa1':
            models_.append(classifiers.PA_I(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa2':
            models_.append(classifiers.PA_II(random_state=seed))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'perceptron':
            models_.append(classifiers.Perceptron(random_state=seed))
            params_.append({})
        if model == 'sop':
            models_.append(classifiers.SecondOrderPerceptron(random_state=seed))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1))
            })
        if model == 'romma':
            models_.append(classifiers.ROMMA(random_state=seed))
            params_.append({})
        if model == 'aromma':
            models_.append(classifiers.aROMMA(random_state=seed))
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
            model_ = __GridSearchCV(model, params_[i], n_jobs=-1)
            model_.fit(X_train, Y_train, verbose=verbose-1)
            # After collecting, let's save, report and proceed
            model.set_params(**model_.best_params_)
            best_params_record += (models[i] + "\n" 
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

        acc = sklearn.metrics.accuracy_score(Y_test, preds)
        f1 = sklearn.metrics.f1_score(Y_test, preds)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Y_test, preds,
                             normalize='true').ravel()

        # ROC would not compute if for instance we have only one class in the
        # test data.
        # This is the case for the svmguide3 dataset bundled with the package.
        try:
            roc = sklearn.metrics.roc_auc_score(Y_test, scores)
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