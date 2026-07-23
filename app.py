"""Servidor Flask para deteccion de fugas en tuberias."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

_classifier = None
_size_estimator = None
_preprocessor = None


def get_models():
    global _classifier, _size_estimator, _preprocessor
    if _classifier is None:
        from pipeline_leak.models.leak_classifier import LeakClassifier
        from pipeline_leak.models.leak_size_estimator import LeakSizeEstimator
        _classifier = LeakClassifier.load("outputs/models/leak_classifier.pkl")
        _size_estimator = LeakSizeEstimator.load("outputs/models/leak_size_estimator.pkl")
        with open("outputs/models/preprocessor.pkl", "rb") as f:
            _preprocessor = pickle.load(f)
    return _classifier, _size_estimator, _preprocessor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        import pandas as pd
        df = pd.DataFrame([{
            "pipeline_type": data.get("pipeline_type", "crude_oil"),
            "pipeline_length_km": float(data.get("pipeline_length_km", 50)),
            "pipeline_diameter_mm": float(data.get("pipeline_diameter_mm", 200)),
            "pressure_upstream_mpa": float(data.get("pressure_upstream_mpa", 5.0)),
            "pressure_downstream_mpa": float(data.get("pressure_downstream_mpa", 4.0)),
            "flow_rate_m3h": float(data.get("flow_rate_m3h", 200)),
            "temperature_c": float(data.get("temperature_c", 25)),
            "ambient_temp_c": float(data.get("ambient_temp_c", 20)),
            "soil_moisture_pct": float(data.get("soil_moisture_pct", 40)),
            "pipe_wall_thickness_mm": float(data.get("pipe_wall_thickness_mm", 12)),
            "pressure_drop_mpa": float(data.get("pressure_drop_mpa", 1.0)),
            "flow_anomaly_m3h": float(data.get("flow_anomaly_m3h", 0)),
            "acoustic_emission_db": float(data.get("acoustic_emission_db", 10)),
            "temperature_diff_c": float(data.get("temperature_diff_c", 1)),
            "vibration_level_g": float(data.get("vibration_level_g", 0.3)),
        }])

        classifier, size_est, preprocessor = get_models()
        X = preprocessor.transform(df)

        has_leak = int(classifier.predict(X)[0])
        proba = classifier.predict_proba(X)[0]

        result = {
            "status": "ok",
            "has_leak": bool(has_leak),
            "leak_probability": round(float(max(proba)), 4),
            "leak_class": "FUGA DETECTADA" if has_leak else "SIN FUGA",
        }

        if has_leak:
            size_pred = float(size_est.predict(X)[0])
            result["leak_severity"] = (
                "critica" if size_pred > 0.8 else
                "alta" if size_pred > 0.5 else
                "media" if size_pred > 0.3 else "baja"
            )
            result["leak_size_score"] = round(size_pred, 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/batch", methods=["POST"])
def api_batch():
    try:
        data = request.json.get("readings", [])
        import pandas as pd
        classifier, _, preprocessor = get_models()

        df = pd.DataFrame(data)
        X = preprocessor.transform(df)
        predictions = classifier.predict(X)
        probas = classifier.predict_proba(X)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probas)):
            results.append({
                "index": i,
                "has_leak": bool(pred),
                "leak_probability": round(float(max(prob)), 4),
            })

        leak_count = sum(1 for r in results if r["has_leak"])
        return jsonify({
            "status": "ok",
            "total_readings": len(results),
            "leaks_detected": leak_count,
            "results": results,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/models")
def api_models():
    classifier, size_est, _ = get_models()
    return jsonify({
        "status": "ok",
        "classifier": classifier.best_name,
        "size_estimator": "trained" if size_est.trained else "not_trained",
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "pipeline-leak-detection"})


if __name__ == "__main__":
    print("=" * 60)
    print("  Servidor Web - Deteccion de Fugas en Tuberias")
    print("=" * 60)
    print("  Cargando modelos...")
    get_models()
    print("  Servidor iniciando en http://127.0.0.1:5005")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5005, debug=True)
