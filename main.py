import argparse
import time
import os

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from olpy.classifiers import *

def print_models():
    models = {
        'alma'  : 'A New Approximate Maximal Margin Classification Algorithm (ALMA)',
        'arow'  : 'Adaptive regularization of weight vectors (AROW)',
        'cw'    : 'Confidence Weighted',
        'scw'   : 'Soft Confidence Weighted',
        'iellip': 'Improved Ellipsoid',
        'narow' : 'New adaptive algorithms for online classification',
        'nherd' : 'Normal Herd',
        'ogd'   : 'Online Gradient Descent',
        'pa'    : 'Passive Aggressive',
        'pa1'   : 'Passive Aggressive I',
        'pa2'   : 'Passive Aggressive II',
        'perceptron': 'Perceptron',
        'sop'   : 'Second Order Perceptron',
        'romma' : 'Relaxed Online Maximum Margin Algorithm',
        'aromma': 'Aggressive ROMMA'
    }

    message = ''
    for short, desc in models.items():
        message += short + "\t\t:" + desc
        message += '\n'

    return message


parser = argparse.ArgumentParser(description='After receiving input\
          from the user, this program train a series of Online Machine\
          Learning models for binary classification.')

parser.add_argument('train_set', metavar='TRAINING', \
             help='file containing the training dataset. CSV file \
                         expected', type=argparse.FileType('r'), nargs=1)
parser.add_argument('test_set', metavar='TESTING', \
                help='containing the test dataset. CSV file \
                        expected', type=argparse.FileType('r'), nargs=1)
parser.add_argument('-l', '--label', type=str, default='Label', \
                    help='index of the target variable.\
                        (default:  %(default)s)')
parser.add_argument('--models', type=str, nargs='+', default='--all',\
                help='The list of models to try from. \n Choices are: \
                        \n' + print_models() + '. or use use %(default)s')
parser.add_argument('-n', type=int, default=1, help='the number of \
                        iterations to run. (default: \
                        %(default)s)')
parser.add_argument('-s', type=int, default=None, help='the random seed\
                        to use in training the models. (default: \
                        %(default)s)')
parser.add_argument('-o', type=str, default='experiment-results.csv',\
                        help='file to which the reports would be saved\
                        (default: %(default)s)')
parser.add_argument('-v', '--verbose', help='whether the program should\
                            have a verbose output or not', action='count'\
                                , default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    # Collect the arguments
    train_file = args.train_set[0]
    test_file = args.test_set[0]
    verbose = args.verbose > 0
    output_file = args.o
    seed = args.s
    models = args.models
    n_iterations = args.n
    label = args.label

    # First we replace all by the list of available models
    if models == '--all' or '--all' in models:
        models = ['alma', 'arow', 'cw', 'scw', 'iellip', 'narow', 'nherd',
                     'ogd', 'pa', 'pa1', 'pa2', 'perceptron', 'sop', 'romma', 'aromma']
    
    # Create a variable to store the model objects
    models_ = []
    for model in set(models):
        model = model.lower()
        if model == 'alma':
            models_.append(ALMA(random_state=seed, num_iterations=n_iterations))
        if model == 'arow':
            models_.append(AROW(random_state=seed, num_iterations=n_iterations))
        if model == 'cw':
            models_.append(CW(random_state=seed, num_iterations=n_iterations))
        if model == 'scw':
            models_.append(SCW(random_state=seed, num_iterations=n_iterations))
        if model == 'iellip':
            models_.append(IELLIP(random_state=seed, num_iterations=n_iterations))
        if model == 'narow':
            models_.append(NAROW(random_state=seed, num_iterations=n_iterations))
        if model == 'nherd':
            models_.append(NHerd(random_state=seed, num_iterations=n_iterations))
        if model == 'ogd':
            models_.append(OGD(random_state=seed, num_iterations=n_iterations))
        if model == 'pa':
            models_.append(PA(random_state=seed, num_iterations=n_iterations))
        if model == 'pa1':
            models_.append(PA_I(random_state=seed, num_iterations=n_iterations))
        if model == 'pa2':
            models_.append(PA_II(random_state=seed, num_iterations=n_iterations))
        if model == 'perceptron':
            models_.append(Perceptron(random_state=seed, num_iterations=n_iterations))
        if model == 'sop':
            models_.append(SecondOrderPerceptron(random_state=seed, num_iterations=n_iterations))
        if model == 'romma':
            models_.append(ROMMA(random_state=seed, num_iterations=n_iterations))
        if model == 'aromma':
            models_.append(aROMMA(random_state=seed, num_iterations=n_iterations))

    # Load the datasets
    scaler = MinMaxScaler()
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    Y_train = train_data.loc[:, label].to_numpy()
    X_train = scaler.fit_transform(train_data.drop(columns=[label]))

    Y_test = test_data.loc[:, label].to_numpy()
    X_test = scaler.fit_transform(test_data.drop(columns=[label]))


    summary = pd.DataFrame(np.zeros((len(models_), 6)), columns=['Training_Time', \
                                    'Prediction_Time', 'Accuracy', 'F1-Score', 'Recall', \
                                        'ROC_AUC-Score'])
    summary.insert(0, 'Model', [model.name for model in models_])

    if verbose:
        print('Algorithm\t\tTrain time (s)\t\tTest time(s)\t\tAccuracy\t\tF1-Score\t\tRecall\t\tROC-AUC')
        print()

    i = 0
    for model in models_:
        training_start = time.time()
        # Try, catch to avoid errors stopping the program
        try:
            model.fit(X_train, Y_train, verbose=False)
            duration = time.time() - training_start

            scores = model.decision_function(X_test)
            test_start = time.time()
            preds = model.predict(X_test)
            preds_duration = time.time() - test_start

            acc = accuracy_score(Y_test, preds)
            f1 = f1_score(Y_test, preds)
            recall = recall_score(Y_test, preds)
            roc = roc_auc_score(Y_test, scores)

            if verbose:
                print("{}\t\t\t{}\t{}\t{}\t{}\t{}\t{}".format(list(set(models))[i][:7], duration, \
                                                    preds_duration, acc, f1, recall, roc))
            summary.loc[i, 'Training_Time'] = duration
            summary.loc[i, 'Prediction_Time'] = preds_duration
            summary.loc[i, 'Accuracy'] = acc
            summary.loc[i, 'F1-Score'] = f1
            summary.loc[i, 'Recall'] = recall
            summary.loc[i, 'ROC_AUC-Score'] = roc
        except Exception as e:
            print(model.name, "- Failed\n", e)
        i = i + 1
    summary.to_csv(output_file, index=False)