import math

import numpy as np


def p_norm(x, p):
    return (sum([x[i] ** p for i in range(len(x))])) ** (1.0/p)