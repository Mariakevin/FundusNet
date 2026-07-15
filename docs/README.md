# FundusNet — AI-Powered Retinal Disease Screening

A production-grade Django web application for automated retinal disease classification using a 5-model deep learning ensemble.

## Overview

FundusNet analyzes retinal fundus images to detect four conditions: **Healthy**, **Cataract**, **Glaucoma**, and **Retina Disease** (Diabetic Retinopathy). The system uses a dual-branch ensemble of CNN and Transformer architectures with Grad-CAM explainability and uncertainty quantification.

### Key Features

- **5-Model Ensemble**: Swin Transformer, MaxViT, ConvNeXt V2, EfficientNet V2, DeiT III
- **Multi-Class Classification**: 4 retinal conditions
- **Grad-CAM Heatmaps**: Visual explanations for predictions
- **Uncertainty Quantification**: MC Dropout for confidence calibration
- **Test-Time Augmentation (TTA)**: Geometric + color augmentations
- **Batch Processing**: Up to 100 images per batch
- **ONNX Runtime**: 3-5x faster inference via ONNX models
- **API Key Authentication**: Configurable per-endpoint auth
- **File-Based Rate Limiting**: 30 requests/minute per IP
- **Dark Mode**: Toggle + system preference detection

## Architecture

```
retina_project/
├── retina_app/                    # Main Django app
│   ├── services/                  # ML services
│   │   ├── inference.py           # Inference orchestrator
│   │   ├── ensemble.py            # 5-model ensemble + stacking
│   │   ├── model_manager.py       # ONNX model loading
│   │   ├── preprocessing.py       # Image preprocessing pipeline
│   │   ├── gradcam.py             # Grad-CAM explainability
│   │   ├── uncertainty.py         # MC Dropout uncertainty
│   │   ├── refusal.py             # Confidence-based refusal
│   │   ├── fundus_validator.py    # Fundus image validation
│   │   ├── image_cache.py         # LRU result caching
│   │   └── batch_inference.py     # Async batch processing
│   ├── templates/                 # HTML templates
│   ├── static/                    # CSS/JS assets
│   ├── api.py                     # REST API endpoints
│   ├── views.py                   # Web views
│   ├── models.py                  # Database models
│   ├── constants.py               # Centralized configuration
│   └── urls.py                    # URL routing
├── retina_project/                # Django settings
│   └── settings/
│       ├── base.py                # Shared settings
│       ├── dev.py                 # Development
│       └── prod.py                # Production
├── models/                        # ONNX model files
├── docs/                          # Documentation
├── gunicorn.conf.py               # Gunicorn config
├── Dockerfile                     # Docker build
└── docker-compose.yml             # Docker Compose
```

## Installation

```bash
# Clone repository
git clone https://github.com/Mariakevin/FundusNet.git
cd FundusNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

## Model Configuration

### Ensemble (5 Models)

| Model | Type | Weight | Purpose |
|-------|------|--------|---------|
| Swin Transformer | Transformer | 0.25 | Best single model |
| MaxViT | Hybrid | 0.25 | Multi-axis attention |
| ConvNeXt V2 | CNN | 0.20 | Strong baseline |
| EfficientNet V2 | CNN | 0.15 | Efficient deployment |
| DeiT III | Transformer | 0.15 | ViT baseline |

### Ensemble Strategy

- **Weighted Averaging**: Per-class dynamic weights via `CLASS_PERFORMANCE_WEIGHTS`
- **Stacking Meta-Learner**: Logistic regression on model outputs (with fallback)
- **Learnable Fusion**: MLP-based dynamic weighting (Res101-MViT-Ens inspired)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | (required) | Django secret key |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated allowed hosts |
| `DJANGO_DEBUG` | `False` | Enable debug mode |
| `FUNDUSNET_API_KEYS` | (empty) | Comma-separated API keys for auth |
| `GUNICORN_BIND` | `0.0.0.0:8000` | Gunicorn bind address |
| `GUNICORN_WORKERS` | `cpu*2+1` | Number of workers |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/` | GET | No | API root |
| `/api/v1/predict/` | POST | Yes | Single image prediction |
| `/api/v1/predict/batch/` | POST | Yes | Batch prediction (async) |
| `/api/v1/jobs/<id>/` | GET | Yes | Job status |
| `/api/v1/health/` | GET | No | Model health |
| `/api/v1/registry/` | GET | No | Model registry |
| `/api/v1/leaderboard/` | GET | No | Model leaderboard |
| `/api/v1/experiments/` | GET | No | Experiment tracking |
| `/api/v1/stats/` | GET | No | Service statistics |

### Authentication

```bash
# Single prediction with API key
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "X-API-Key: your-key-here" \
  -F "image=@retinal_image.jpg"
```

## Web Views

| Route | Description |
|-------|-------------|
| `/` | Single image upload and analysis |
| `/history/` | Prediction history |
| `/export/<id>/?format=json\|csv` | Export prediction |
| `/batch/` | Multi-file batch upload |

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or with environment variables
DJANGO_SECRET_KEY=your-secret-key \
DJANGO_ALLOWED_HOSTS=yourdomain.com \
docker-compose up --build
```

## Testing

```bash
# Run all tests
python manage.py test retina_app

# Run specific test file
python manage.py test retina_app.tests.test_api
```

## Security

- API key authentication on prediction endpoints
- File-based rate limiting (30 req/min per IP)
- Path traversal protection on media files
- MIME type validation
- CSP headers (HSTS, X-Frame-Options: DENY)
- Non-root Docker user
- Soft-delete for audit trail

## License

Proprietary — All rights reserved
