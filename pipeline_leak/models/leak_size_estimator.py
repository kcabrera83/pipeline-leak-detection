import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


class LeakSizeEstimator:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        )
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        pred = self.model.predict(X)
        return {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "r2": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred),
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
