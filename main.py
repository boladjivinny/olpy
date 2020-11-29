import argparse
import time
import os
import joblib

import pandas as pd
import numpy as np

from olpy.classifiers import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight


class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


def print_models():
    list_of_models = {
        'alma': 'A New Approximate Maximal Margin Classification Algorithm (ALMA)',
        'arow': 'Adaptive regularization of weight vectors (AROW)',
        'cw': 'Confidence Weighted',
        'scw': 'Soft Confidence Weighted',
        'scw2': 'Soft Confidence Weighted (version 2)',
        'iellip': 'Improved Ellipsoid',
        'narow': 'New adaptive algorithms for online classification',
        'nherd': 'Normal Herd',
        'ogd': 'Online Gradient Descent',
        'pa': 'Passive Aggressive',
        'pa1': 'Passive Aggressive I',
        'pa2': 'Passive Aggressive II',
        'perceptron': 'Perceptron',
        'sop': 'Second Order Perceptron',
        'romma': 'Relaxed Online Maximum Margin Algorithm',
        'aromma': 'Aggressive ROMMA'
    }

    message = ''
    for short, desc in list_of_models.items():
        message += short + "\t\t:" + desc
        message += '\n'

    return message


def olpy_parse_args():
    parser = argparse.ArgumentParser(description='After receiving input\
              from the user, this program train a series of Online Machine\
              Learning models for binary classification.')

    parser.add_argument('train_set', metavar='TRAINING',
                        help='file containing the training dataset. CSV file \
                             expected', type=argparse.FileType('r'), nargs=1)
    parser.add_argument('test_set', metavar='TESTING',
                        help='containing the test dataset. CSV file \
                            expected', type=argparse.FileType('r'), nargs=1)
    parser.add_argument('-l', '--label', type=str, default='Label',
                        help='index of the target variable.\
                            (default:  %(default)s)')
    parser.add_argument('--models', type=str, nargs='+', default='--all',
                        help='The list of models to try from. \n Choices are: \
                            \n' + print_models() + '. or use use %(default)s')
    parser.add_argument('-n', type=int, default=1, help='the number of \
                            iterations to run. (default: \
                            %(default)s)')
    parser.add_argument('-s', type=int, default=None, help='the random seed\
                            to use in training the models. (default: \
                            %(default)s)')
    parser.add_argument('-o', type=str, default='experiment-results.csv',
                        help='file to which the reports would be saved\
                            (default: %(default)s)')
    parser.add_argument('-b', '--bias', help="Whether or not a bias should be \
                                             used.", action="store_true")

    parser.add_argument('-d', '--dump', help="Whether or not a the models \
                                             should be dumped.", action="store_true")

    parser.add_argument('-w', '--use-weights', help="Whether or not a the models \
                                                 should be dumped.", action="store_true")

    parser.add_argument('--dump-dir', action=FullPaths, type=is_dir, default='.',
                        help="Output directory for dumping the models."
                             '(default: %(default)s)')

    parser.add_argument('--cv', help="Whether or not hyper-parameter through \
                                     cross validation should be done.", action="store_true")

    parser.add_argument('-v', '--verbose', help='whether the program should \
                                have a verbose output or not', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = olpy_parse_args()
    # Collect the arguments
    train_file = args.train_set[0]
    test_file = args.test_set[0]
    verbose = args.verbose
    output_file = args.o
    seed = args.s
    models = args.models
    n_iterations = args.n
    label = args.label
    bias = args.bias
    use_weights = args.use_weights
    cv = args.cv
    dump = args.dump
    model_dir = args.dump_dir

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

    class_weight = None
    if use_weights:
        class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)

    # First we replace all by the list of available models
    if models == '--all' or '--all' in models:
        models = ['alma', 'arow', 'cw', 'scw', 'scw2', 'iellip', 'narow', 'nherd',
                  'ogd', 'pa', 'pa1', 'pa2', 'perceptron', 'sop', 'romma', 'aromma']

    # Create a variable to store the model objects
    models_ = []
    params_ = []

    for model in set(models):
        model = model.lower()
        if model == 'alma':
            models_.append(ALMA(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'B': list(np.arange(0.1, 1, 0.1)),
                'p': range(2, 12, 2),
                'alpha': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'arow':
            models_.append(AROW(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'r': [2 ** i for i in range(-4, 5)]
            })
        if model == 'cw':
            models_.append(CW(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'a': list(np.arange(0.1, 1, 0.1)),
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw':
            models_.append(SCW(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'scw2':
            models_.append(SCW2(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)],
                'eta': list(np.arange(0.50, 1, 0.05))
            })
        if model == 'iellip':
            models_.append(IELLIP(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'b': list(np.arange(0.1, 1.1, 0.1)),
                'c': list(np.arange(0.1, 1.1, 0.1))
            })
        if model == 'narow':
            models_.append(NAROW(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
            })
        if model == 'nherd':
            models_.append(NHerd(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1)),
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'ogd':
            models_.append(OGD(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa':
            models_.append(PA(random_state=seed, num_iterations=n_iterations))
            params_.append({})
        if model == 'pa1':
            models_.append(PA_I(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'pa2':
            models_.append(PA_II(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'C': [2 ** i for i in range(-4, 5)]
            })
        if model == 'perceptron':
            models_.append(Perceptron(random_state=seed, num_iterations=n_iterations))
            params_.append({})
        if model == 'sop':
            models_.append(SecondOrderPerceptron(random_state=seed, num_iterations=n_iterations))
            params_.append({
                'a': list(np.arange(0.1, 1.1, 0.1))
            })
        if model == 'romma':
            models_.append(ROMMA(random_state=seed, num_iterations=n_iterations))
            params_.append({})
        if model == 'aromma':
            models_.append(aROMMA(random_state=seed, num_iterations=n_iterations))
            params_.append({})

    summary = pd.DataFrame(np.zeros((len(models_), 10)), columns=['Training-Time', 'Prediction-Time', 'Accuracy',
                                                                  'F1-Score', 'Recall', 'ROC_AUC-Score', 'FP', 'FN',
                                                                  'TP', 'TN'])
    summary.insert(0, 'Model', [model.name for model in models_])

    if verbose:
        print("%9s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s" %
              ('algorithm', 'train time (s)', 'test time (s)', 'accuracy', 'f1-score', 'roc-auc',
               'true positive', 'true negative', 'false positive', 'false negative'))
        print()

    i = 0
    best_params_record = "Best params: \n"
    for model in models_:
        model_ = model
        if use_weights:
            model_.set_params(class_weight=class_weight)
        if cv:
            model_ = GridSearchCV(model, params_[i], refit='recall', n_jobs=-1)
        training_start = time.time()
        # Try, catch to avoid errors stopping the program
        try:
            model_.fit(X_train, Y_train, verbose=False)
            duration = time.time() - training_start

            scores = model_.decision_function(X_test)
            test_start = time.time()
            preds = model_.predict(X_test)
            preds_duration = time.time() - test_start

            acc = accuracy_score(Y_test, preds)
            f1 = f1_score(Y_test, preds)
            tn, fp, fn, tp = confusion_matrix(Y_test, preds, normalize='true').ravel()

            # ROC would not compute if for instance we have only one class in the test dataset.
            # This is the case for svmguide3 dataset included
            try:
                roc = roc_auc_score(Y_test, scores)
            except ValueError:
                roc = 0

            if cv:
                best_params_record += model.name + "\n" + str(model_.best_params_) + "\n\n"

            if verbose:
                print("%-12s\t%-3f\t%-3f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f\t%-5f" %
                      (list(set(models))[i], 1000 * duration, 1000 * preds_duration, acc, f1,
                       roc, tp, tn, fp, fn))

            summary.loc[i, 'Training-Time'] = duration
            summary.loc[i, 'Prediction-Time'] = preds_duration
            summary.loc[i, 'Accuracy'] = acc
            summary.loc[i, 'F1-Score'] = f1
            summary.loc[i, 'ROC_AUC-Score'] = roc
            summary.loc[i, 'TP'] = tp
            summary.loc[i, 'TN'] = tn
            summary.loc[i, 'FP'] = fp
            summary.loc[i, 'FN'] = fn

            if dump:
                # Save the model
                joblib.dump(model_, model_dir + '/' + list(set(models))[i] + '.dump')

        except Exception as e:
            print(e)
            print(model.name, "- Failed\n", e)
        i = i + 1
    if cv:
        print()
        print()
        print(best_params_record)
    if output_file:
        summary.to_csv(output_file, index=False)
