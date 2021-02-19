import numpy as np 
import pandas as pd


class Dataset:
    """A helper class to load various datasets.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        train_data (:obj:`pandas.DataFrame`): The training data without
            the labels.
        train_target (:obj:`numpy.ndarray`): The output variable for the
            training data.
        test_data (:obj:`pandas.DataFrame`): The testing data without 
            the labels.
        test_target (:obj:`numpy..ndarray`): The output variable for the
            test data.

    Args:
        f_train (str): The path to the file containing the training
            dataset.
        f_test (str): The path to the file containing the testing
            dataset.
        label (str, optional): The column in which the target variable
            is located in the files. Defaults to `Label`.

    Raises:
        FileNotFoundError: if the supplied files are inexistent.
        IndexError: if the label provided does not match any column in
            the file.

    """

    def __init__(self, f_train, f_test, label='Label'):
        # Load the datasets
        self.train_data = pd.read_csv(f_train)
        self.test_data = pd.read_csv(f_test)

        # Retrieve the relevant parts
        self.train_target = self.train_data[label].to_numpy()
        self.train_data = self.train_data.drop(columns=[label]).to_numpy()
        self.test_target = self.test_data[label].to_numpy()
        self.test_data = self.test_data.drop(columns=[label]).to_numpy()