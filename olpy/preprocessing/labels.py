import numpy as np

from olpy.exceptions import NotFittedError


class LabelEncoder:
    """Encodes an output vector to match the specifications.

    Given that online learning algorithms usually work on output 
    vectors with entries (-1, 1), this function performs this action
    for the user.

    Attributes:
        y (:obj:`array` of `ndarray`): the data to be transformed.
        positive_label (:obj:`int`, optional): The number in the output
            field that represents the positive label. The value passed
            should be different than -1. Defaults to 1.
        labels (:obj:`tuple`): represents the labels that are present
            in the dataset. This can be used at prediction time.
    """
    def __init__(self, positive_label=1):
        self.y = None
        self.positive_label = positive_label
        self.labels = None

    def fit(self, y):
        """Fits the output vector y.

        This method parses the parsed value and sets the necessary 
        values to transform it later.

        Args:
            y (:obj:`list` or `numpy.ndarray`): the data to be transformed.

        Returns:
            self: the current instance.

        Raises:
            ValueError: if the number of labels is different than 2.
            AssertionError: if the positive label is not found in the
                labels.
        """
        self.y = y
        labels = np.unique(self.y)

        # First check that we have two values
        if len(labels) != 2:
            raise ValueError(
                'Expected two labels. Got {} instead'.format(len(labels))
                )

        # Let's check now that the specified positive label is in the array
        assert self.positive_label in labels,\
             'The positive label ({}) has not been found in the labels'
             
        self.labels = (
            set(labels).difference(
                {self.positive_label}
                ).pop(), self.positive_label)

        return self

    def transform(self, return_labels=True):
        """Transforms the data.

        Based on the information collected while fitting, this fuction
        returns the transformed labels that can be used directly for
        training.

        Args:
            return_labels (bool, optional): whether the labels should
                be returned or not. Default `True`.

        Returns:
            `numpy.ndarray` if return_labels is True else None

        Raises:
            NotFittedError if the encoder was not already fitted.
        """
        # All is okay. Now we can change
        if self.y is None:
            raise NotFittedError(
                    None, 
                    None, 
                    'Attempted to transform an unfitted encorder.'
                )
        if return_labels:
            return (
                np.array(
                    [1 if self.y[i] == self.positive_label 
                    else -1 for i in range(self.y.shape[0])]), 
                self.labels
            )
        else:
            return np.array(
                [1 if self.y[i] == self.positive_label 
                else -1 for i in range(self.y.shape[0])])

    def fit_transform(self, y, return_labels=True):
        """Fits and transforms the data.

        Combines the actions of `fit` and `transform` methods.

        Args:
            return_labels (:obj:`bool`, optional): whether the labels should
                be returned or not. Default `True`.
            y (:obj:`array` of `ndarray`): the data to be transformed.

        Returns:
            `numpy.ndarray` if return_labels is True else None

        Raises:
            ValueError: if the number of labels is different than 2.
            AssertionError: if the positive label is not found in the
                    labels.
            NotFittedError if the encoder was not already fitted.

        """
        self.fit(y)
        return self.transform(return_labels=return_labels)
