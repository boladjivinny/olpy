import numpy as np
from sklearn.exceptions import NotFittedError as SKLearnNotFittedError
from olpy.exceptions import NotFittedError

class Querier:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.best_index = 0

    def fetch(self, models):
        # This one should return the index of the best one
        pass

    def __next__(self):
        if self.X.shape[0] == 0: raise StopIteration()
        # Get the next best
        x, y = self.X[self.best_index], self.y[self.best_index]
        # delete them from there
        self.X = np.delete(self.X, obj=self.best_index, axis=0)
        self.y = np.delete(self.y, obj=self.best_index, axis=0)
        return x, y


class CommiteeQuerier(Querier):
    def fetch(self, models):
        # We first predict using the models
        results = []
        for model in models:
            model_prediction = None
            try:
                model_prediction = model.predict(self.X)
            except (NotFittedError, SKLearnNotFittedError):
                model_prediction = [0] * self.X.shape[0]
            except AttributeError:
                model_prediction = [0 if not model.predict_one(x) else model.predict_one(x) for x in self.X]
            finally:
                results.append((np.where(model_prediction == 0, -1, model_prediction)).tolist())

        # Now we have the prediction of each model for the dataset
        # Let us keep a count now. We want 
        results = np.array(results)
        #print(results)
        # We will use the absolute value to ease things on us
        counts = [abs(sum(preds)) for preds in results.T]
        # In case there is a lot of divergence, the value should be as low as possible
        self.best_index = np.argmin(counts)
        
        print(self.best_index)
