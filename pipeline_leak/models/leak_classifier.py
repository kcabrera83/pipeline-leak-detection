import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from sklearn.metrics import accuracy_score
import pickle


class LeakClassifier:
    def __init__(self):
        self.detectors = {
            "iforest": IForest(contamination=0.05, random_state=42),
            "knn": KNN(contamination=0.05),
            "hbos": HBOS(contamination=0.05),
        }
        self.best_model = None
        self.best_name = None
        self.results = {}

    def train(self, X, y):
        for name, det in self.detectors.items():
            det.fit(X)
            raw_labels = det.labels_
            mapped = np.where(raw_labels == 1, 1, 0)
            acc = accuracy_score(y, mapped)
            self.results[name] = {"accuracy": acc}

        best_name = max(self.results, key=lambda k: self.results[k]["accuracy"])
        self.best_model = self.detectors[best_name]
        self.best_name = best_name
        return self.results

    def predict(self, X):
        raw = self.best_model.predict(X)
        return np.where(raw == 1, 1, 0)

    def predict_proba(self, X):
        scores = self.best_model.decision_scores_
        min_s, max_s = scores.min(), scores.max()
        norm = (scores - min_s) / (max_s - min_s + 1e-8)
        X_scores = self.best_model.decision_function(X)
        X_norm = (X_scores - min_s) / (max_s - min_s + 1e-8)
        X_norm = np.clip(X_norm, 0, 1)
        proba_no_leak = 1 - X_norm
        proba_leak = X_norm
        return np.column_stack([proba_no_leak, proba_leak])

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {"accuracy": accuracy_score(y, pred)}

    def feature_importance(self):
        if hasattr(self.best_model, "feature_importances_"):
            return self.best_model.feature_importances_
        return None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
