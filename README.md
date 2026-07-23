# Pipeline Leak Detection

ML system for oil and gas pipeline leak detection using pressure, flow, temperature, vibration, and acoustic emission sensors.

## Models

- **Leak Classifier**: Random Forest / Gradient Boosting to classify whether a leak exists (yes/no)
- **Leak Size Estimator**: Gradient Boosting to estimate leak severity

## Usage

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Dashboard: http://127.0.0.1:5005

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web dashboard |
| POST | `/api/predict` | Detect individual leak |
| POST | `/api/batch` | Batch leak detection |
| GET | `/api/models` | Model info |
| GET | `/api/health` | Health check |

## Model Features

- Upstream/downstream pressure and flow rate
- Acoustic emission and vibration
- Temperature difference
- Soil moisture and pipeline type

Elaborado por Ing. Kelvin Cabrera
