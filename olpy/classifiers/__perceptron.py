import numpy as np
import numpy.linalg as LA

from olpy._model import Model, BCModelWithLabelEncoding


class Perceptron(BCModelWithLabelEncoding):
    def __init__(self, X, y):
        super().__init__(X, y)
    
    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            f_t     = self.weights.dot(x_t)
            hat_y_t = 1 if f_t >= 0 else -1
            if y_t != hat_y_t:
                self.weights    = self.weights + y_t * x_t
        return hat_y_t

class SecondOrderPerceptron(Perceptron):
    def __init__(self, X, y, a=1):
        super().__init__(X, y)
        self.Sigma  = a * np.eye(X.shape[1])
    
    def fit(self):
        for x_t, y_t in zip(self.X, self.y):
            S_x_t   = x_t @ self.Sigma
            v_t     = x_t @ S_x_t.T 
            beta_t  = 1 / (v_t + 1)
            Sigma_t = self.Sigma - beta_t * S_x_t.T @ S_x_t

            f_t     = self.weights @ Sigma_t @ x_t.T
            hat_y_t = 1 if f_t >= 0 else -1
            
            if y_t != hat_y_t:
                self.weights    = self.weights + y_t * x_t
        return hat_y_t

    def predict(self, data):
        S_x     = data @ self.Sigma
        v       = data @ S_x.T 
        beta    =  LA.inv(v + np.ones(data.shape[0], data.shape[1]))
        Sigma   = self.Sigma - beta @ S_x.T @ S_x
        
        pred    = self.weights @ Sigma @ data.T
        
        return [self.labels[0] if x < 0 else 1 for x in pred]