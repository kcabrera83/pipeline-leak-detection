# User Guide - Pipeline Leak Detection

## Overview
ML system for oil and gas pipeline leak detection using pressure, flow, temperature, vibration, and acoustic emission sensors. Classifies leak presence and estimates leak severity.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/pipeline-leak-detection.git
cd pipeline-leak-detection
pip install -r requirements.txt
```

### Training Models
```bash
python train.py
```
Generates 5000 synthetic pipeline records, trains leak classifier and size estimator.

### Starting the Server
```bash
python app.py
```
Open http://localhost:5005 in your browser.

## Dashboard Features
- **Leak Detection Form**: 15-field form for real-time leak detection
- **Batch Analysis**: Analyze multiple pipeline readings at once
- **Severity Assessment**: Leak size estimation with severity classification
- **Model Information**: View trained model details

## Pipeline Parameters

### Input Parameters (15 features)
| Parameter | Unit | Description |
|-----------|------|-------------|
| pipeline_type | - | Type of fluid (crude_oil, natural_gas, refined, water) |
| pipeline_length_km | km | Pipeline segment length |
| pipeline_diameter_mm | mm | Pipe internal diameter |
| pressure_upstream_mpa | MPa | Upstream pressure |
| pressure_downstream_mpa | MPa | Downstream pressure |
| flow_rate_m3h | m3/h | Flow rate |
| temperature_c | deg C | Fluid temperature |
| ambient_temp_c | deg C | Ambient temperature |
| soil_moisture_pct | % | Soil moisture content |
| pipe_wall_thickness_mm | mm | Pipe wall thickness |
| pressure_drop_mpa | MPa | Pressure differential |
| flow_anomaly_m3h | m3/h | Flow rate anomaly |
| acoustic_emission_db | dB | Acoustic emission level |
| temperature_diff_c | deg C | Temperature differential |
| vibration_level_g | g | Vibration level |

### Output
- **has_leak**: Boolean (true/false)
- **leak_probability**: Confidence score (0-1)
- **leak_class**: "LEAK DETECTED" or "NO LEAK"
- **leak_severity**: low / medium / high / critical
- **leak_size_score**: Continuous severity score (0-1)

## API Usage

### Using curl
```bash
# Single leak detection
curl -X POST http://localhost:5005/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Batch analysis
curl -X POST http://localhost:5005/api/batch \
  -H "Content-Type: application/json" \
  -d '{"readings": [{"pipeline_type": "crude_oil", "pipeline_length_km": 50, ...}, ...]}'
```

### Using Python
```python
import requests

# Single leak detection
response = requests.post("http://localhost:5005/api/predict", json={
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
})
result = response.json()
print(f"Leak: {result['has_leak']} ({result['leak_probability']:.1%})")
if result['has_leak']:
    print(f"Severity: {result['leak_severity']}")

# Batch analysis
readings = [{"pipeline_type": "crude_oil", ...}, ...]
response = requests.post("http://localhost:5005/api/batch", json={"readings": readings})
batch = response.json()
print(f"Detected {batch['leaks_detected']} leaks in {batch['total_readings']} readings")
```

## Models
- **Leak Classifier**: GradientBoosting/Random Forest for binary leak detection
- **Leak Size Estimator**: GradientBoosting for leak severity scoring
