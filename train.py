"""Entrenamiento para deteccion de fugas en tuberias."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_leak.data_generator import PipelineDataGenerator
from pipeline_leak.utils.preprocessor import PipelinePreprocessor
from pipeline_leak.models.leak_classifier import LeakClassifier
from pipeline_leak.models.leak_size_estimator import LeakSizeEstimator


def main():
    print("=" * 60)
    print("  Entrenamiento - Deteccion de Fugas en Tuberias")
    print("=" * 60)

    print("\n[1/5] Generando datos sinteticos...")
    gen = PipelineDataGenerator(n_samples=5000, random_state=42)
    df = gen.save("outputs/data/pipeline_data.csv")
    print(f"  Dataset: {len(df)} registros, {len(df.columns)} columnas")

    print("\n[2/5] Preprocesando datos...")
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

    print("\n[3/5] Entrenando clasificador de fugas...")
    classifier = LeakClassifier()
    cls_results = classifier.train(X_train, y_leak_train)
    cls_eval = classifier.evaluate(X_test, y_leak_test)
    print(f"  Mejor modelo: {classifier.best_name}")
    print(f"  Train Accuracy: {cls_results[classifier.best_name]['accuracy']:.4f}")
    print(f"  Test  Accuracy: {cls_eval['accuracy']:.4f}")

    print("\n[4/5] Entrenando estimador de tamano de fuga...")
    size_est = LeakSizeEstimator()
    size_results = size_est.train(X_size_train, y_size_train)
    size_eval = size_est.evaluate(X_size_test, y_size_test)
    print(f"  Train R2: {size_results['r2']:.4f} | MAE: {size_results['mae']:.4f}")
    print(f"  Test  R2: {size_eval['r2']:.4f} | MAE: {size_eval['mae']:.4f}")

    print("\n[5/5] Guardando modelos...")
    os.makedirs("outputs/models", exist_ok=True)
    classifier.save("outputs/models/leak_classifier.pkl")
    size_est.save("outputs/models/leak_size_estimator.pkl")
    import pickle
    with open("outputs/models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("  Modelos guardados en outputs/models/")

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Leak Classifier:    Accuracy={cls_eval['accuracy']:.4f}")
    print(f"  Leak Size Estimator: R2={size_eval['r2']:.4f} | MAE={size_eval['mae']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
