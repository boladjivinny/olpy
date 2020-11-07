import numpy as np
from olpy._model import OnlineLearningModel


class AROW(OnlineLearningModel):
    # Need to check the r parameter 
    def __init__(self, r=1, num_iterations=20, random_state=None, positive_label=1):
        """
        Instantiate an AROW model for training.
        
        Crammer, K.; Kulesza, A. & Dredze, M.
        Adaptive regularization of weight vectors 
        Advances in neural information processing systems, 2009, 414-422

        Parameters
        ----------
        r   : float, default 1
            AROW's parameter, r > 0
        num_iterations: int
            Represents the number of iterations to run the algorithm.
        random_state:   int, default None
            Seed for the pseudorandom generator
        positive_label: 1 or -1
            Represents the value that is used as positive_label.

        Returns
        -------
        None
        """
        super().__init__(num_iterations=num_iterations, random_state=random_state, positive_label=positive_label)
        self.r = r
        self.sigma = None

    def _update(self, x: np.ndarray, y: int):
        f_t = self.weights.dot(x)
        v_t = x @ self.sigma @ x.T
        loss = max(0, 1 - f_t * y)
        if loss > 0:
            beta_t = 1 / (v_t + self.r)
            alpha_t = loss * beta_t
            sigma = np.expand_dims(x @ self.sigma.T, axis=0)
            self.weights += alpha_t * y * np.squeeze(sigma)
            self.sigma -= beta_t * sigma.T @ sigma

    def _setup(self, X: np.ndarray, Y: np.ndarray):
        self.sigma = np.identity(X.shape[1])

    def get_params(self):
        {'r': self.r, 'num_iterations': self.num_iterations}
