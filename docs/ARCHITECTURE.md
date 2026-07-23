# Architecture - Pipeline Leak Detection

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5005       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
+-------------v-----------+  +-------------v-----------+
| Leak Classifier         |  | Leak Size Estimator     |
| (Binary Classification) |  | (Severity Regression)   |
| GB/RF                   |  | GradientBoosting        |
+-------------+-----------+  +-------------+-----------+
              |                             |
+-------------v-----------------------------v-----------+
|              PipelinePreprocessor                     |
|       (Encoding + Scaling)                            |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (5000 records)     |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic pipeline data generator (`PipelineDataGenerator`) producing 5000 records
- **Pipeline Types**: crude_oil, natural_gas, refined, water
- **Sensor Data**: 15 features (pressure, flow, temperature, vibration, acoustic, soil)
- **Preprocessing**: Standard scaling, one-hot encoding for pipeline type

### Model Layer

#### Leak Classifier
- **Algorithm**: GradientBoosting (best) among Random Forest, SVM, KNN, MLP
- **Target**: Binary leak detection (has_leak: 0/1)
- **Features**: 15 pipeline and sensor features
- **Output**: Binary prediction + probability score
- **Metrics**: Accuracy, Precision, Recall, F1

#### Leak Size Estimator
- **Algorithm**: GradientBoosting
- **Target**: Leak severity score (continuous, based on pressure drop)
- **Training Data**: Only samples with confirmed leaks (has_leak=1)
- **Output**: Continuous severity score (0-1)
- **Severity Mapping**: low (<=0.3), medium (<=0.5), high (<=0.8), critical (>0.8)

### API Layer
- **Framework**: Flask
- **Endpoints**: 5 REST endpoints (health, models, predict, batch, docs)
- **Batch Processing**: Supports multiple readings in single request
- **Model Loading**: Lazy loading with pickle deserialization

### Dashboard Layer
- **Frontend**: Flask + HTML/CSS/JS
- **Features**: Leak detection form, batch analysis, severity visualization

## Data Flow

1. **Input**: 15 pipeline/sensor readings
2. **Preprocessing**: `PipelinePreprocessor` encodes pipeline type and scales features
3. **Classification**: LeakClassifier determines leak presence (binary)
4. **Severity Estimation**: If leak detected, LeakSizeEstimator scores severity
5. **Response**: Leak status, probability, severity level, and size score

## Batch Processing Flow
1. **Input**: Array of pipeline readings
2. **Preprocessing**: Batch transform all readings
3. **Classification**: Classify all readings in parallel
4. **Aggregation**: Count total leaks, return individual results
5. **Response**: Summary + per-reading results

## Training Pipeline
1. Generate synthetic pipeline data (5000 records)
2. Split data into leak/no-leak subsets
3. Train leak classifier on full dataset (80/20 split)
4. Train leak size estimator on leak-only samples
5. Evaluate accuracy, R2, MAE
6. Save models to `outputs/models/`

## File Structure
```
pipeline-leak-detection/
├── pipeline_leak/
│   ├── data_generator.py       # Synthetic pipeline data
│   ├── models/
│   │   ├── leak_classifier.py  # Binary leak detection
│   │   └── leak_size_estimator.py # Severity estimation
│   └── utils/
│       └── preprocessor.py     # Data preprocessing
├── outputs/models/             # Trained models
├── templates/index.html        # Dashboard
├── app.py                      # Flask server
├── train.py                    # Training pipeline
└── test_api.py                 # API tests
```
