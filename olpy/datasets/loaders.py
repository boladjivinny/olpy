import pkg_resources
from pathlib import Path

from . dataset import Dataset

def load_a1a():
    """"Loads the `a1a` dataset for usage in a program.

    Args:
        None

    Returns:
        An object of class `Dataset` with the a1a dataset loaded.

    Example:
        >>> from olpy.datasets import load_a1a
        >>> a1a = load_a1a()
        >>> a1a.train_data.columns
        Index(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        ...
        '114', '115', '116', '117', '118', '119', '120', '121', '122', '123'],
        dtype='object', length=123)
    """
    i_stream = pkg_resources.resource_stream(__name__, 'data/a1a')
    o_stream = pkg_resources.resource_stream(__name__, 'data/a1a.t')

    return Dataset(i_stream, o_stream, '0')

def load_svmguide1():
    """"Loads the `svmguide1` dataset for usage in a program.

    Args:
        None

    Returns:
        An object of class `Dataset` with the svmguide1 dataset loaded.

    Example:
        >>> from olpy.datasets import load_svmguide1
        >>> svmguide1 = load_svmguide1()
        >>> svmguide1.train_data.columns
        Index(['1', '2', '3', '4'], dtype='object')
    """
    i_stream = pkg_resources.resource_stream(__name__, 'data/svmguide1')
    o_stream = pkg_resources.resource_stream(__name__, 'data/svmguide1.t')

    return Dataset(i_stream, o_stream, '0')

def load_svmguide3():
    """"Loads the `svmguide3` dataset for usage in a program.

    Args:
        None

    Returns:
        An object of class `Dataset` with the svmguide3 dataset loaded.

    Example:
        >>> from olpy.datasets import load_svmguide3
        >>> svmguide3 = load_svmguide3()
        >>> svmguide3.train_data.columns
        Index(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
        '14', '15', '16', '17', '18', '19', '20', '21'],
        dtype='object')
    """
    i_stream = pkg_resources.resource_stream(__name__, 'data/svmguide3')
    o_stream = pkg_resources.resource_stream(__name__, 'data/svmguide3.t')

    return Dataset(i_stream, o_stream, '0')