# API Documentation - Pipeline Leak Detection

## Base URL
```
http://localhost:5005
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with leak detection panels.

---

### GET /api/health
Service health check.

**Response (200):**
```json
{"status": "ok", "service": "pipeline-leak-detection"}
```

---

### GET /api/models
Information about trained models.

**Response (200):**
```json
{
  "status": "ok",
  "classifier": "GradientBoosting",
  "size_estimator": "trained"
}
```

---

### POST /api/predict
Detect leaks in a pipeline using 15 sensor readings.

**Request:**
```json
{
  "pipeline_type": "crude_oil",
  "pipeline_length_km": 50,
  "pipeline_diameter_mm": 200,
  "pressure_upstream_mpa": 5.0,
  "pressure_downstream_mpa": 4.0,
  "flow_rate_m3h": 200,
  "temperature_c": 25,
  "ambient_temp_c": 20,
  "soil_moisture_pct": 40,
  "pipe_wall_thickness_mm": 12,
  "pressure_drop_mpa": 1.0,
  "flow_anomaly_m3h": 0,
  "acoustic_emission_db": 10,
  "temperature_diff_c": 1,
  "vibration_level_g": 0.3
}
```

**Response (200) - No Leak:**
```json
{
  "status": "ok",
  "has_leak": false,
  "leak_probability": 0.1200,
  "leak_class": "NO LEAK"
}
```

**Response (200) - Leak Detected:**
```json
{
  "status": "ok",
  "has_leak": true,
  "leak_probability": 0.9500,
  "leak_class": "LEAK DETECTED",
  "leak_severity": "high",
  "leak_size_score": 0.6500
}
```

**Leak Severity Levels:**
- `low`: score <= 0.3
- `medium`: 0.3 < score <= 0.5
- `high`: 0.5 < score <= 0.8
- `critical`: score > 0.8

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### POST /api/batch
Batch analysis of multiple pipeline readings.

**Request:**
```json
{
  "readings": [
    {
      "pipeline_type": "crude_oil",
      "pipeline_length_km": 50,
      "pipeline_diameter_mm": 200,
      "pressure_upstream_mpa": 5.0,
      "pressure_downstream_mpa": 4.0,
      "flow_rate_m3h": 200,
      "temperature_c": 25,
      "ambient_temp_c": 20,
      "soil_moisture_pct": 40,
      "pipe_wall_thickness_mm": 12,
      "pressure_drop_mpa": 1.0,
      "flow_anomaly_m3h": 0,
      "acoustic_emission_db": 10,
      "temperature_diff_c": 1,
      "vibration_level_g": 0.3
    },
    {
      "pipeline_type": "natural_gas",
      ...
    }
  ]
}
```

**Response (200):**
```json
{
  "status": "ok",
  "total_readings": 2,
  "leaks_detected": 1,
  "results": [
    {"index": 0, "has_leak": false, "leak_probability": 0.1500},
    {"index": 1, "has_leak": true, "leak_probability": 0.9200}
  ]
}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - invalid input or processing error |
| 500 | Internal server error |
