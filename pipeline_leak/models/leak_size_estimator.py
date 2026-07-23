import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


class LeakSizeEstimator:
    def __init__(self):
        self.model = AutoEncoder(
            contamination=0.05,
            hidden_neurons=[64, 32, 16, 32, 64],
            epochs=50,
            batch_size=32,
            random_state=42,
            verbose=0,
        )
        self.trained = False
        self._regressor_weights = None

    def train(self, X, y):
        self.model.fit(X)
        self.trained = True

        recon = self.model.decision_scores_
        min_s, max_s = recon.min(), recon.max()
        self._regressor_weights = (min_s, max_s)

        pred = self._reconstruct_to_score(X)
        return {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}

    def _reconstruct_to_score(self, X):
        scores = self.model.decision_function(X)
        min_s, max_s = self._regressor_weights
        normalized = (scores - min_s) / (max_s - min_s + 1e-8)
        return np.clip(normalized, 0, 1)

    def predict(self, X):
        return self._reconstruct_to_score(X)

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
