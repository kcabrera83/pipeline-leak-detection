import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


class LeakClassifier:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=2024, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=2024,
            ),
        }
        self.best_model = None
        self.best_name = None
        self.results = {}

    def train(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
            train_pred = model.predict(X)
            acc = accuracy_score(y, train_pred)
            self.results[name] = {"accuracy": acc}

        best_name = max(self.results, key=lambda k: self.results[k]["accuracy"])
        self.best_model = self.models[best_name]
        self.best_name = best_name
        return self.results

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        return None

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
