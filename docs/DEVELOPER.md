# Developer Guide

Internal documentation for FundusNet development and maintenance.

---

## Project Structure

```
retina_project/
├── retina_app/                    # Main Django app
│   ├── services/                  # ML services
│   │   ├── inference.py           # Inference orchestrator
│   │   ├── ensemble.py            # 5-model ensemble + stacking
│   │   ├── model_manager.py       # ONNX model loading
│   │   ├── preprocessing.py       # Image preprocessing pipeline
│   │   ├── fundus_validator.py    # Fundus image validation
│   │   ├── gradcam.py             # Grad-CAM explainability
│   │   ├── uncertainty.py         # MC Dropout uncertainty
│   │   ├── refusal.py             # Confidence-based refusal
│   │   ├── image_cache.py         # LRU result caching
│   │   ├── batch_inference.py     # Async batch processing
│   │   └── exceptions.py          # Custom exceptions
│   ├── ml/                        # ML utilities
│   │   └── registry.py            # Model registry + experiment tracking
│   ├── static/retina_app/
│   │   └── medical.css            # Frontend styles
│   ├── templates/                 # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── history.html
│   │   ├── batch.html
│   │   └── errors/                # Error pages (400, 403, 404, 500)
│   ├── tests/                     # Test suite
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
├── models/                        # ONNX model files (gitignored)
├── media/                         # Uploaded files (runtime)
├── docs/                          # Documentation
├── gunicorn.conf.py               # Gunicorn config
├── Dockerfile                     # Docker build
├── docker-compose.yml             # Docker Compose
└── requirements.txt               # Python dependencies
```

---

## Development Workflow

### Running the Server

```bash
# Development
python manage.py runserver

# Production (with gunicorn config)
gunicorn -c gunicorn.conf.py retina_project.wsgi:application

# Docker
docker-compose up --build
```

### Database Operations

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Show migration status
python manage.py showmigrations
```

### Loading Models

Place ONNX model files in `models/` directory:
- `swin_retinopathy.onnx`
- `maxvit_retinopathy.onnx`
- `convnext_v2_retinopathy.onnx`
- `efficientnet_v2_retinopathy.onnx`
- `deit_retinopathy.onnx`

---

## API Authentication

API key authentication protects prediction endpoints. Configure via environment:

```bash
# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in environment
export FUNDUSNET_API_KEYS="your-api-key-here"
```

### Using the API

```bash
# Single prediction (with API key)
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "X-API-Key: your-api-key-here" \
  -F "image=@retinal_image.jpg"

# Single prediction (without API key, if open access)
curl -X POST http://localhost:8000/api/v1/predict/ \
  -F "image=@retinal_image.jpg"

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch/ \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"]}'

# Health check (no auth required)
curl http://localhost:8000/api/v1/health/
```

---

## Rate Limiting

Rate limiting is file-based (works across Gunicorn workers):

- **Default**: 30 requests per minute per IP
- **Storage**: `/tmp/fundusnet_ratelimit/` (configurable)
- **Response**: 429 status with `Retry-After` header

---

## Testing

```bash
# Run all tests
python manage.py test retina_app

# Run specific test file
python manage.py test retina_app.tests.test_api

# Run with coverage
coverage run manage.py test retina_app
coverage report
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | (required) | Django secret key |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated allowed hosts |
| `DJANGO_DEBUG` | `False` | Enable debug mode |
| `FUNDUSNET_API_KEYS` | (empty) | Comma-separated API keys |
| `GUNICORN_BIND` | `0.0.0.0:8000` | Gunicorn bind address |
| `GUNICORN_WORKERS` | `cpu*2+1` | Number of workers |

---

## Model Configuration

All model configuration is centralized in `constants.py`:

- `MODEL_LIST` — List of model names
- `MODEL_WEIGHTS` — Ensemble weights (must sum to 1.0)
- `MODEL_NAME_MAP` — Mapping to timm model identifiers
- `CLASS_PERFORMANCE_WEIGHTS` — Per-class dynamic weights
- `CATEGORIES` — Classification labels

---

## Troubleshooting

### Models Not Loading

1. Check ONNX files exist in `models/` directory
2. Verify `onnxruntime` is installed: `pip install onnxruntime`
3. Check logs for specific model errors

### Rate Limiting Issues

1. Clear rate limit files: `rm /tmp/fundusnet_ratelimit/*.json`
2. Adjust limits in `api.py`: `FileRateLimiter(max_requests=60)`

### Memory Issues

1. Reduce `MAX_WORKERS` in constants
2. Reduce `MAX_CACHE_SIZE` for less memory usage
3. Use fewer models in ensemble
