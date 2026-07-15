# Developer Guide

Internal documentation for FundusNet development and maintenance.

---

## Project Structure

```
retina_project/
├── retina_app/               # Main Django application
│   ├── ml/                   # ML utilities (registry only)
│   ├── services/             # Core services
│   │   ├── inference.py      # Inference orchestrator
│   │   ├── ensemble.py       # Multi-model ensemble + uncertainty
│   │   ├── model_manager.py  # ONNX model loading
│   │   ├── preprocessing.py  # Image preprocessing
│   │   ├── fundus_validator.py # Fundus image validation
│   │   ├── gradcam.py        # Grad-CAM explainability
│   │   ├── image_cache.py    # Result caching
│   │   └── exceptions.py     # Custom exceptions
│   ├── static/
│   │   └── retina_app/
│   │       ├── medical.css   # Frontend styles
│   │       └── medical.js    # Frontend JavaScript
│   ├── templates/            # HTML templates
│   ├── tests/                # Test suite
│   ├── api.py                # REST API endpoints
│   ├── constants.py          # Centralized configuration
│   ├── models.py             # Database models
│   ├── urls.py               # URL routing
│   └── views.py              # View functions
├── retina_project/           # Django project settings
│   ├── settings/
│   │   ├── base.py           # Base configuration
│   │   ├── dev.py            # Development settings
│   │   └── prod.py           # Production settings
│   ├── urls.py               # Project URLs
│   └── wsgi.py               # WSGI application
├── docs/                     # Documentation
├── models/                   # ONNX model files (gitignored)
├── media/                    # Uploaded files (runtime)
├── gunicorn.conf.py          # Gunicorn configuration
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose
├── manage.py                 # Django management
└── requirements.txt          # Python dependencies
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

# Reset database
python manage.py migrate retina_app zero
python manage.py migrate
```

### Loading Models

Place ONNX model files in `models/` directory:
- `swin_retinopathy.onnx` - Swin Transformer
- `maxvit_retinopathy.onnx` - MaxViT
- `convnext_v2_retinopathy.onnx` - ConvNeXt V2
- `efficientnet_v2_retinopathy.onnx` - EfficientNet V2
- `deit_retinopathy.onnx` - DeiT III

---

## API Authentication

API key authentication protects prediction endpoints. Configure via environment:

```bash
# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in environment
export FUNDUSNET_API_KEYS="your-api-key-here"

# Or in .env file
FUNDUSNET_API_KEYS=your-api-key-here
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
| `DJANGO_PRODUCTION` | `False` | Enable production mode |
| `FUNDUSNET_API_KEYS` | (empty) | Comma-separated API keys |
| `FUNDUSNET_MEDIA_ROOT` | `media/` | Media storage path |
| `FUNDUSNET_MODELS_DIR` | `models/` | ONNX models directory |
| `GUNICORN_BIND` | `0.0.0.0:8000` | Gunicorn bind address |
| `GUNICORN_WORKERS` | `cpu*2+1` | Number of workers |
| `GUNICORN_THREADS` | `4` | Threads per worker |

---

## Model Configuration

All model configuration is centralized in `constants.py`:

- `MODEL_LIST` - List of model names
- `MODEL_WEIGHTS` - Ensemble weights
- `MODEL_NAME_MAP` - Mapping to timm model identifiers
- `CLASS_PERFORMANCE_WEIGHTS` - Per-class weights

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
