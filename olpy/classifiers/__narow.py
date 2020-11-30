import numpy as np
from olpy._model import OnlineLearningModel


class NAROW(OnlineLearningModel):
    name = "NAROW"
    
    def __init__(self, a=1, num_iterations=20, random_state=None,
                 positive_label=1, class_weight=None):
        """
        Instantiate a new NAROW model for training.
        
        Orabona, F. & Crammer, K.
        New adaptive algorithms for online classification 
        Advances in Neural Information Processing Systems, 
        Curran Associates, Inc., 2010, 23, 1840-1848

        Parameters
        ----------
        a   : float, default 1
            NAROW's parameter, a > 0
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
        super().__init__(num_iterations=num_iterations, random_state=random_state,
                         positive_label=positive_label, class_weight=class_weight)
        self.a = a

    def _update(self, x: np.ndarray, y: int):
        decision = self.weights.dot(x)
        v_t = x @ self.sigma @ x.T
        m_t = decision * y
        if 1 - m_t > 0:
            if v_t > (1 / self.a):
                r_t = v_t / (self.a * v_t - 1)
            else:
                r_t = -float("inf")
            beta_t = 1 / (v_t + r_t)
            alpha_t = max(0, 1 - m_t) * beta_t * self.class_weight_[y]
            sigma = np.expand_dims(x @ self.sigma.T, axis=0)
            self.weights += alpha_t * y * np.squeeze(sigma)
            self.sigma -= beta_t * sigma.T @ sigma

    def _setup(self, X: np.ndarray):
        self.sigma = np.identity(X.shape[1])

    def get_params(self, deep=True):
        params = super().get_params()
        params['a'] = self.a

        return params
