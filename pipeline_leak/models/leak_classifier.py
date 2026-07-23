"""Modelo de clasificacion de fugas."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


class LeakClassifier:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            ),
        }
        self.best_model = None
        self.best_name = None
        self.results = {}

    def train(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            self.results[name] = {"accuracy": acc}

        best_name = max(self.results, key=lambda k: self.results[k]["accuracy"])
        self.best_model = self.models[best_name]
        self.best_name = best_name
        return self.results

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, pred),
            "report": classification_report(y, pred, output_dict=True),
        }

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
