"""Tests de API para pipeline-leak-detection."""

import sys
import json

sys.path.insert(0, ".")
from app import app

client = app.test_client()
PASSED = 0
FAILED = 0


def test(name, method, url, body=None, expect_status=200):
    global PASSED, FAILED
    try:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=body)
        data = json.loads(resp.data)
        ok = resp.status_code == expect_status and data.get("status") == "ok"
        if ok:
            PASSED += 1
        else:
            FAILED += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({resp.status_code})")
    except Exception as e:
        FAILED += 1
        print(f"  [FAIL] {name}: {e}")


def main():
    print("=" * 60)
    print("  Tests - Pipeline Leak Detection")
    print("=" * 60)
    test("GET /api/health", "GET", "/api/health")
    test("GET /api/models", "GET", "/api/models")
    test("POST /api/predict (no leak)", "POST", "/api/predict", {
        "pipeline_type": "crude_oil", "pipeline_length_km": 50, "pipeline_diameter_mm": 200,
        "pressure_upstream_mpa": 5, "pressure_downstream_mpa": 4.5, "flow_rate_m3h": 200,
        "temperature_c": 25, "ambient_temp_c": 20, "soil_moisture_pct": 40,
        "pipe_wall_thickness_mm": 12, "pressure_drop_mpa": 0.5, "flow_anomaly_m3h": 0,
        "acoustic_emission_db": 8, "temperature_diff_c": 0.5, "vibration_level_g": 0.2,
    })
    print(f"\n  Resultado: {PASSED}/{PASSED+FAILED} tests pasaron")
    print("=" * 60)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
