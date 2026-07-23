# Pipeline Leak Detection

ML system for oil and gas pipeline leak detection using PyOD ensemble methods, scipy statistical analysis, and FastAPI.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Anomaly Detection | **PyOD** - IForest, KNN, HBOS ensemble |
| Statistical Analysis | **scipy** - Statistical tests and distributions |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- PyOD - Isolation Forest, KNN, and HBOS for anomaly detection
- scipy - Statistical analysis and hypothesis testing
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Models

- **Leak Classifier**: PyOD ensemble (IForest, KNN, HBOS) for leak detection
- **Leak Size Estimator**: Gradient Boosting for leak severity estimation

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

## Detection Methods

| Method | Type | Description |
|--------|------|-------------|
| IForest | Isolation Forest | Tree-based anomaly isolation |
| KNN | K-Nearest Neighbors | Distance-based anomaly scoring |
| HBOS | Histogram-Based | Feature-wise anomaly scoring |

---

Elaborado por Ing. Kelvin Cabrera
