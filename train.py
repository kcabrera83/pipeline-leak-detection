import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_leak.data_generator import PipelineDataGenerator
from pipeline_leak.utils.preprocessor import PipelinePreprocessor
from pipeline_leak.models.leak_classifier import LeakClassifier
from pipeline_leak.models.leak_size_estimator import LeakSizeEstimator
from scipy import stats


def grubbs_test(data, threshold=0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return False
    grubbs_stat = max(abs(data - mean)) / std
    t_crit = stats.t.ppf(1 - threshold / (2 * n), n - 2)
    grubbs_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return grubbs_stat > grubbs_crit


def main():
    print("=" * 60)
    print("  Pipeline Leak Detection - PyOD + scipy.stats")
    print("=" * 60)

    print("\n[1/5] Generating synthetic data...")
    gen = PipelineDataGenerator(n_samples=5000, random_state=42)
    df = gen.save("outputs/data/pipeline_data.csv")
    print(f"  Dataset: {len(df)} records, {len(df.columns)} columns")

    print("\n[2/5] Preprocessing data...")
    preprocessor = PipelinePreprocessor()
    X = preprocessor.fit_transform(df)
    y_leak = df["has_leak"].values
    y_size = df.loc[df["has_leak"] == 1, "pressure_drop_mpa"].values
    X_leak = X[df["has_leak"] == 1]

    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_leak_train, y_leak_test = y_leak[:split], y_leak[split:]

    split_size = int(0.8 * len(X_leak))
    X_size_train, X_size_test = X_leak[:split_size], X_leak[split_size:]
    y_size_train, y_size_test = y_size[:split_size], y_size[split_size:]

    print(f"  Train: {split} | Test: {len(df) - split}")
    print(f"  Leak samples: {len(X_leak)} (train: {split_size}, test: {len(X_leak) - split_size})")

    print("\n[3/5] Training leak classifier (PyOD ensemble)...")
    classifier = LeakClassifier()
    cls_results = classifier.train(X_train, y_leak_train)
    cls_eval = classifier.evaluate(X_test, y_leak_test)
    print(f"  Best detector: {classifier.best_name}")
    print(f"  Train Accuracy: {cls_results[classifier.best_name]['accuracy']:.4f}")
    print(f"  Test  Accuracy: {cls_eval['accuracy']:.4f}")

    print("\n[4/5] Training leak size estimator (PyOD AutoEncoder)...")
    size_est = LeakSizeEstimator()
    size_results = size_est.train(X_size_train, y_size_train)
    size_eval = size_est.evaluate(X_size_test, y_size_test)
    print(f"  Train R2: {size_results['r2']:.4f} | MAE: {size_results['mae']:.4f}")
    print(f"  Test  R2: {size_eval['r2']:.4f} | MAE: {size_eval['mae']:.4f}")

    print("\n  Statistical outlier detection (Grubbs test on pressure_drop)...")
    pressure_data = df["pressure_drop_mpa"].values
    is_outlier = grubbs_test(pressure_data, threshold=0.05)
    print(f"  Grubbs test result: {'Outliers detected' if is_outlier else 'No significant outliers'}")
    print(f"  Mean: {np.mean(pressure_data):.4f} | Std: {np.std(pressure_data):.4f}")

    print("\n[5/5] Saving models...")
    os.makedirs("outputs/models", exist_ok=True)
    classifier.save("outputs/models/leak_classifier.pkl")
    size_est.save("outputs/models/leak_size_estimator.pkl")
    import pickle
    with open("outputs/models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("  Models saved to outputs/models/")

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Leak Classifier:     Accuracy={cls_eval['accuracy']:.4f}")
    print(f"  Leak Size Estimator: R2={size_eval['r2']:.4f} | MAE={size_eval['mae']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
