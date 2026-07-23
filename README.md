# Pipeline Leak Detection

Sistema ML para deteccion de fugas en tuberias de petroleo y gas usando sensores de presion, flujo, temperatura, vibracion y emision acustica.

## Modelos

- **Leak Classifier**: Random Forest / Gradient Boosting para clasificar si hay fuga (si/no)
- **Leak Size Estimator**: Gradient Boosting para estimar severidad de la fuga

## Uso

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Dashboard: http://127.0.0.1:5005

### Endpoints

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/` | Dashboard web |
| POST | `/api/predict` | Detectar fuga individual |
| POST | `/api/batch` | Detectar fugas en lote |
| GET | `/api/models` | Info de modelos |
| GET | `/api/health` | Health check |

## Features del Modelo

- Presion upstream/downstream y caudal
- Emision acustica y vibracion
- Diferencia de temperatura
- Humedad del suelo y tipo de tuberia

Elaborado por Ing. Kelvin Cabrera
