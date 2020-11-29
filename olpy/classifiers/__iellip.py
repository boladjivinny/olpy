import numpy as np
import math

from olpy import OnlineLearningModel


class IELLIP(OnlineLearningModel):
    name = "Improved Ellipsoid"
    
    def __init__(self, a=1, b=0.3, c=0.1, num_iterations=20, \
                random_state=None, positive_label=1, class_weight=None):
        """
        Instantiate an IELLIP model for training.

        This function creates an instance of the IELLIP online learning
        algorithm.
        
        Yang, L.; Jin, R. & Ye, J., Online learning by ellipsoid method, 
        Proc. 26th Annu. Int. Conf. Machine Learning, Association for 
        Computing Machinery, 2009, 1153-1160

        Parameters
        ----------
        a   : float, default 1
            Trade-off parameter. a is in the range [0,1]
        b,c : float, default b=0.3, c=0.1
            Parameters controlling the memory of online learning.
            0 <= b, c <=1
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.
        class_weight: dict
            Represents the relative weight of the labels in the dataset.
            Useful for imbalanced classification tasks.

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, \
                            positive_label=positive_label, class_weight=class_weight)
        self.a = a
        self.b = b
        self.c = c
        self.sigma = None

    def _update(self, x, y):
        decision = self.weights.dot(x)
        prediction = np.sign(decision)
        v = (x @ self.sigma @ x.T) * self.class_weight_[y]
        m = y * decision
        if prediction != y:
            if v != 0:
                alpha = (1 - m) / math.sqrt(v)
                g = y * x / math.sqrt(v)
                # Added an axis to avoid errors
                sigma = np.expand_dims(g @ self.sigma.T, axis=0)
                self.weights = self.weights + alpha * np.squeeze(sigma)
                self.sigma = (self.sigma - self.c * sigma.T @ sigma) / (1 - self.c)
        self.c *= self.b

    def _setup(self, X: np.ndarray):
        self.sigma = self.a * np.eye(X.shape[1])

    def get_params(self, deep=True):
        params = super().get_params()

        params['a'] = self.a
        params['b'] = self.b
        params['c'] = self.c

        return params