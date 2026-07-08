# Developer Guide

Internal documentation for FundusNet development and maintenance.

---

## Project Structure

```
retina_project/
├── retina_app/               # Main Django application
│   ├── migrations/            # Database migrations
│   ├── services/
│   │   ├── inference.py     # ML inference engine
│   │   └── exceptions.py     # Custom exceptions
│   ├── static/
│   │   └── retina_app/
│   │       ├── medical.css   # Maximalist design system
│   │       └── medical.js    # Frontend JavaScript
│   ├── templates/            # HTML templates
│   ├── admin.py             # Django admin configuration
│   ├── forms.py             # Form classes
│   ├── models.py            # Database models
│   ├── test_inference.py    # ML inference tests
│   ├── urls.py              # URL routing
│   └── views.py             # View functions
├── retina_project/           # Django project settings
│   ├── settings/
│   │   ├── base.py          # Base configuration
│   │   ├── dev.py          # Development settings
│   │   └── prod.py         # Production settings
│   ├── urls.py             # Project URLs
│   └── wsgi.py             # WSGI application
├── docs/                    # Documentation
├── media/                   # Uploaded files (runtime)
├── train.py                 # Training script
├── manage.py               # Django management
└── *.pth                   # Model weights files
```

---

## Development Workflow

### Running the Server

```bash
# Development
python manage.py runserver

# With specific port
python manage.py runserver 8000

# Production (requires gunicorn)
gunicorn retina_project.wsgi:application
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

Place trained model files in project root:
- `squeezenet_retinopathy.pth` - SqueezeNet weights
- `efficientnet_retinopathy.pth` - EfficientNet weights
- `resnet_retinopathy.pth` - ResNet50 weights
- `mobilenet_retinopathy.pth` - MobileNet weights

---

## Adding New Models

### Step 1: Add Model Configuration

In `services/inference.py`:

```python
MODEL_LIST.append("newmodel")
MODEL_WEIGHTS["newmodel"] = 0.15
MODEL_VERSIONS["newmodel"] = "newmodel-retinopathy-v1"
```

### Step 2: Implement Loading Logic

In `_load_model_with_checkpoint`:

```python
elif model_type == "newmodel":
    model = models.newmodel(weights=None)
    in_features = model.fc.in_features
    model.fc = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
```

### Step 3: Update Classifier Creation

In `_create_improved_classifier`, add handling for new architecture.

---

## Adding New Endpoints

### Step 1: Define View Function

In `views.py`:

```python
@login_required
def new_view(request):
    # View logic here
    return render(request, "template.html", context)
```

### Step 2: Add URL Route

In `retina_app/urls.py`:

```python
path('new-endpoint/', views.new_view, name='new_view'),
```

### Step 3: Create Template (if needed)

Create `templates/new_view.html`

---

## Custom Exception Handling

### Creating New Exception

In `services/exceptions.py`:

```python
class NewException(FundusNetError):
    status_code = 400
    default_message = "New error occurred"

    def __init__(self, message=None):
        self.message = message or self.default_message
        super().__init__(self.message)
```

### Handling in Views

```python
try:
    # Operation that may raise exception
    pass
except NewException as exc:
    logger.warning(f"New error: {exc}")
    return render(request, "error.html", {"error": str(exc)})
```

---

## Template Development

### Adding New Template

1. Create HTML file in `templates/`
2. Extend base template:

```django
{% extends "base.html" %}
{% block content %}
<!-- Your content -->
{% endblock %}
```

### Using Static Files

```django
{% load static %}
<link rel="stylesheet" href="{% static 'retina_app/medical.css' %}">
<script src="{% static 'retina_app/medical.js' %}"></script>
```

---

## Testing

### Running Tests

```bash
# All tests
python manage.py test retina_app

# Specific test file
python manage.py test retina_app.test_inference

# With verbose output
python manage.py test -v 2 retina_app
```

### Writing New Tests

In `test_inference.py` or new test file:

```python
from django.test import TestCase
from retina_app.services.inference import predict_image

class InferenceTestCase(TestCase):
    def test_prediction_returns_label(self):
        result = predict_image("test_image.jpg")
        self.assertIn("label", result)
        self.assertIn(result["label"], ["Normal", "Diseased"])
```

---

## Performance Tuning

### Model Loading

- Models are lazy-loaded via `ModelManager`
- First request may be slower
- Subsequent requests use cached models

### Image Caching

- LRU cache for processed images
- Max 100 items, ~50MB limit
- Auto-clears on memory pressure

### Database Queries

- Use `select_related()` for foreign keys
- Use `prefetch_related()` for reverse FK
- Add indexes for frequently filtered fields

---

## Logging

### Application Logs

Logger name: `retina_app`

Configuration in `settings/base.py`:

```python
LOGGING = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'retina_app.log',
        },
    },
    'loggers': {
        'retina_app': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    },
}
```

### Log Levels

- `DEBUG`: Detailed debug info
- `INFO`: General operation info
- `WARNING`: Potential issues
- `ERROR`: Failures requiring attention
- `CRITICAL`: Critical system failures

---

## Troubleshooting

### Common Issues

**Model not loading:**
- Check file exists in project root
- Verify file format (.pth)
- Check file permissions

**Slow inference:**
- Normal on first request (model loading)
- Check image cache status
- Monitor memory usage

**Database errors:**
- Run migrations: `python manage.py migrate`
- Check database connection settings

**Authentication issues:**
- Clear session: `python manage.py clearsessions`
- Check SECRET_KEY consistency

---

## Code Style

- Follow PEP 8 for Python
- Use type hints where beneficial
- Add docstrings to public functions
- Keep lines under 100 characters
- Use meaningful variable names

### Import Order

```python
# Standard library
import os
import logging

# Third party
import torch
import cv2

# Django
from django.shortcuts import render

# Local
from retina_app.models import PredictionRecord
from retina_app.services.inference import predict_image
```

---

## Security Notes

- Never log passwords or secrets
- Validate all user inputs
- Use parameterized database queries (Django ORM)
- Sanitize file uploads
- Keep dependencies updated

---

## Maintenance

### Regular Tasks

- [ ] Monitor error logs weekly
- [ ] Check disk space for media uploads
- [ ] Update dependencies monthly
- [ ] Backup database regularly
- [ ] Review security patches

### Cleanup Commands

```bash
# Clear session table
python manage.py clearsessions

# Clear cache
python manage.py shell -c "from django.core.cache import cache; cache.clear()"

# Delete old media files (custom management command needed)
```

---

## Version History

See `CHANGELOG.md` for detailed version information.
