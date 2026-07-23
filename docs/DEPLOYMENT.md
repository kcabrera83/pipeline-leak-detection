# Deployment Guide - Pipeline Leak Detection

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5005

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t pipeline-leak-detection .
docker run -p 5005:5005 pipeline-leak-detection
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5005:5005"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5005 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/pipeline-leak-detection.git
cd pipeline-leak-detection
pip install -r requirements.txt
python train.py
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5005 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Add authentication for API endpoints
- Validate pipeline parameter ranges

### Monitoring
- Monitor `/api/health` for uptime
- Track leak detection rates
- Alert on high false-positive rates
- Log all predictions for auditing

### Performance
- Batch endpoint processes multiple readings efficiently
- Pre-load models at startup
- Consider async processing for large batch requests

### Critical Safety
- Leak detection is safety-critical; ensure high recall
- Monitor model performance degradation
- Keep backup detection systems
- Regular model retraining with new data

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5005/api/docs`
