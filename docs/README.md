# FundusNet - Medical Retina Screening System

A production-grade Django-based medical image classification system for automated diabetic retinopathy screening using deep learning ensemble models.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Model System](#model-system)
6. [API Endpoints](#api-endpoints)
7. [Data Models](#data-models)
8. [Security](#security)
9. [Production Deployment](#production-deployment)

---

## Overview

FundusNet is a web-based medical screening application that analyzes retinal fundus images to detect potential diabetic retinopathy. The system uses an ensemble of deep learning models (SqueezeNet, EfficientNet, ResNet, MobileNet) with test-time augmentation for improved accuracy.

### Key Features

- **Binary Classification**: Normal vs Diseased
- **Ensemble Inference**: Multiple models with weighted averaging
- **Test-Time Augmentation (TTA)**: 8 augmentations for robust predictions
- **CLAHE Preprocessing**: Contrast enhancement for fundus images
- **Rate Limiting**: 10 requests per minute per user
- **Soft Delete**: Data preservation with soft-delete mechanism
- **Batch Processing**: Handle up to 10 images simultaneously

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Django Application                      │
├─────────────────────────────────────────────────────────────┤
│  Views Layer (views.py)                                      │
│  ├── home_view - Dashboard with statistics                   │
│  ├── classify_retinopathy - Image upload & classification   │
│  ├── history_view - Prediction history with filters          │
│  └── batch_classify - Bulk processing                        │
├─────────────────────────────────────────────────────────────┤
│  Services Layer                                              │
│  ├── inference.py - ML model management & inference          │
│  ├── exceptions.py - Custom error handling                   │
│  └── (models.py, forms.py, admin.py)                        │
├─────────────────────────────────────────────────────────────┤
│  Models Layer                                                │
│  ├── UserProfile - Extended user information                │
│  ├── UploadedImage - Raw uploaded images                    │
│  └── PredictionRecord - Classification results              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ML Inference Engine                         │
├─────────────────────────────────────────────────────────────┤
│  Model Architecture                                          │
│  ├── ModelManager - Lazy loading & caching                  │
│  ├── Ensemble Prediction - Weighted model averaging        │
│  ├── TTA Pipeline - 8 augmentations per image               │
│  └── Preprocessing - CLAHE, ROI detection, quality check    │
├─────────────────────────────────────────────────────────────┤
│  Models Available                                            │
│  ├── SqueezeNet (weight: 0.4) - Primary model              │
│  ├── EfficientNet-B0 (weight: 0.3)                          │
│  ├── ResNet50 (weight: 0.2)                                 │
│  └── MobileNetV3 (weight: 0.1)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9+
- Django 4.2+
- PyTorch 2.0+
- OpenCV (cv2)
- Pillow

### Setup

```bash
# Navigate to project directory
cd retina_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Debug mode | `True` |
| `SECRET_KEY` | Django secret key | Required for production |
| `ALLOWED_HOSTS` | Allowed hostnames | `localhost,127.0.0.1` |
| `MEDIA_ROOT` | Uploaded files directory | `media/` |
| `MAX_UPLOAD_SIZE` | Max upload size in MB | `10` |

### Settings Files

- **Development**: `settings/dev.py`
- **Production**: `settings/prod.py`
- **Base**: `settings/base.py`

---

## Model System

### Supported Models

The system uses an ensemble of 4 CNN architectures:

| Model | Architecture | Parameters | Primary Use |
|-------|-------------|------------|-------------|
| SqueezeNet | Lightweight CNN | 1.2M | Fast inference |
| EfficientNet-B0 | Efficient compound | 5.3M | Best accuracy |
| ResNet50 | Residual network | 25.6M | Robust baseline |
| MobileNetV3 | Mobile-optimized | 2.5M | Edge deployment |

### Ensemble Strategy

```python
MODEL_WEIGHTS = {
    "squeezenet": 0.4,
    "efficientnet": 0.3,
    "resnet": 0.2,
    "mobilenet": 0.1,
}
```

### Test-Time Augmentation (TTA)

8 augmentations are applied during inference:
1. Standard resize (224x224)
2. Horizontal flip
3. 90° rotation
4. 270° rotation
5. Scale 90%
6. Scale 110%
7. Augmented transform (brightness, contrast, rotation)
8. Additional augmented transform

### Image Preprocessing Pipeline

1. **ROI Detection**: Circular fundus region detection using Hough transform
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Green Channel Extraction**: Enhanced blood vessel contrast
4. **Quality Assessment**: Blur, brightness, contrast, saturation scoring
5. **Normalization**: ImageNet mean/std normalization

---

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/accounts/login/` | GET, POST | User login |
| `/accounts/register/` | GET, POST | User registration |
| `/accounts/logout/` | POST | User logout |

### Main Application

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard with statistics |
| `/classify/` | GET, POST | Image upload & classification |
| `/history/` | GET | Prediction history |
| `/history/delete/<id>/` | POST | Delete single prediction |
| `/history/delete-all/` | POST | Bulk delete |
| `/history/reanalyze/<id>/` | POST | Re-run prediction |

### Async Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/check-status/<job_id>/` | GET | Poll job status |
| `/batch-classify/` | POST | Bulk classification |

---

## Data Models

### UserProfile

```python
class UserProfile(models.Model):
    user = OneToOneField(User)
    organization = CharField(max_length=100)
    role = CharField(choices=ROLE_CHOICES)
    phone = CharField(max_length=20)
    created_at = DateTimeField
    updated_at = DateTimeField
```

### UploadedImage

```python
class UploadedImage(models.Model):
    user = ForeignKey(User)
    image = ImageField(upload_to="uploads/")
    uploaded_at = DateTimeField(auto_now_add=True)
```

### PredictionRecord

```python
class PredictionRecord(models.Model):
    user = ForeignKey(User)
    uploaded_image = ForeignKey(UploadedImage, null=True)
    patient_identifier = CharField(max_length=50)  # MRN
    clinical_notes = TextField
    predicted_class = CharField  # "Normal" or "Diseased"
    confidence = FloatField  # 0.0 - 1.0
    model_version = CharField
    created_at = DateTimeField
    is_deleted = BooleanField(default=False)  # Soft delete
```

### Database Indexes

- `user`, `-uploaded_at` on UploadedImage
- `user`, `-created_at` on PredictionRecord
- `predicted_class` on PredictionRecord

---

## Security

### Authentication & Authorization

- Django's built-in authentication system
- `@login_required` decorator on all protected views
- Object-level permission checks for predictions

### Rate Limiting

```python
@rate_limit(max_requests=10, period=60)  # 10 requests per 60 seconds
```

### Production Security Settings (settings/prod.py)

- **HSTS**: Enabled with 1-year max age
- **Content Security Policy**: Strict script/style-src
- **Referrer Policy**: strict-origin-when-cross-origin
- **Permissions Policy**: geolocation=(), microphone=()
- **Secure Cookies**: HttpOnly, Secure, SameSite
- **CSRF Protection**: Enabled
- **X-Frame-Options**: DENY

### File Security

- Path traversal protection on media files
- User ownership verification for downloads
- MIME type validation
- File size limits (10MB max)

---

## Production Deployment

### Required Environment Variables

```bash
SECRET_KEY=<django-secret-key>
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
DATABASE_URL=postgres://user:pass@localhost:5432/fundusnet
MEDIA_ROOT=/var/www/fundusnet/media
```

### Checklist

- [ ] Set `DEBUG=False` in production
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use strong `SECRET_KEY`
- [ ] Set up PostgreSQL database
- [ ] Configure static file serving (nginx/Apache)
- [ ] Set up HTTPS/SSL
- [ ] Configure logging
- [ ] Set up backups
- [ ] Configure monitoring

### Performance Optimization

1. **Model Caching**: Models loaded once, cached in ModelManager
2. **Image Caching**: LRU cache for processed images
3. **ThreadPoolExecutor**: Parallel model inference (2 workers)
4. **Database Indexes**: Optimized query performance

---

## Testing

Run tests with:

```bash
python manage.py test retina_app
```

### Test Coverage

- **Inference Tests** (`test_inference.py`): 16 tests covering:
  - Exception handling
  - Model configuration
  - Image validation
  - Transform pipelines

### Test Categories

1. Exception tests: InferenceError, ImageValidationError, ImageCorruptError, ImageSizeError
2. Category tests: Binary classification (Normal/Diseased)
3. Model config tests: Model loading, weights, ensemble settings
4. Validation config tests: Size limits, dimension constraints
5. Transform tests: ImageNet normalization parameters

---

## Error Handling

### Custom Exceptions (services/exceptions.py)

| Exception | HTTP Status | User Message |
|-----------|-------------|---------------|
| InferenceError | 500 | "Prediction failed. Please try again." |
| ImageValidationError | 400 | "Invalid image file." |
| ImageCorruptError | 400 | "Image appears corrupted." |
| ImageSizeError | 400 | "Image file too large/small." |
| ImageDimensionError | 400 | "Image dimensions out of range." |
| ModelLoadError | 503 | "Model unavailable. Try again later." |
| PreprocessingError | 500 | "Image processing failed." |
| PredictionTimeoutError | 504 | "Prediction timed out." |

---

## License

Proprietary - All rights reserved

---

## Support

For issues and questions, contact the development team.