# API Reference

Detailed documentation of all Django views and their endpoints.

---

## Authentication Views

### Login View (`login_view`)

**URL**: `/accounts/login/`

**Methods**: GET, POST

**GET Response**: Renders `login.html` with LoginForm

**POST Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| username | string | Yes | User's username |
| password | string | Yes | User's password |
| remember_me | boolean | No | Keep session for 30 days |

**POST Success**: Redirects to `home` view

**POST Failure**: Re-renders form with errors

---

### Register View (`register_view`)

**URL**: `/accounts/register/`

**Methods**: GET, POST

**GET Response**: Renders `register.html` with RegisterForm

**POST Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| username | string | Yes | Unique username |
| email | string | Yes | Valid email address |
| password1 | string | Yes | Password |
| password2 | string | Yes | Password confirmation |
| organization | string | No | Medical organization |
| role | string | No | Professional role |

**POST Success**: Auto-login and redirect to `home`

---

### Logout View (`logout_view`)

**URL**: `/accounts/logout/`

**Method**: POST only

**Response**: Redirects to login page

---

## Main Application Views

### Home View (`home_view`)

**URL**: `/` (index)

**Method**: GET

**Authentication**: Required

**Context Variables**:
```python
{
    "total_analyses": int,           # Total predictions
    "healthy_count": int,            # Normal predictions
    "abnormal_count": int,           # Diseased predictions
    "prediction_breakdown": list,   # Class distribution
    "recent_analyses": list,         # Last 10 predictions
    "monthly_count": int,            # This month's predictions
    "avg_confidence": float,         # Average confidence
    "last_login": datetime,         # User's last login
    "all_predictions": list,         # Last 50 predictions
}
```

---

### Classify Retinopathy (`classify_retinopathy`)

**URL**: `/classify/`

**Methods**: GET, POST

**Authentication**: Required

**Rate Limit**: 10 requests per 60 seconds

**POST Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | file | Yes | Fundus image |
| async | string | No | "true" for async processing |

**POST Success Response** (sync):
```python
{
    "prediction": "Normal" | "Diseased",
    "confidence": 85.5,           # Percentage
    "model_version": "ensemble-v4-models-tta",
    "image_url": "/media/uploads/...",
    "prob_labels": [
        ("Normal", 85.5),
        ("Diseased", 14.5)
    ]
}
```

**POST Async Response**:
Redirects to `processing.html` with job_id

**Error Responses**:
- 400: Invalid image
- 429: Rate limit exceeded
- 503: Model unavailable

---

### Check Prediction Status (`check_prediction_status`)

**URL**: `/check-status/<job_id>/`

**Method**: GET

**Authentication**: Required

**Response** (JSON):
```python
// Processing
{"status": "processing", "elapsed": "5.2s"}

// Completed
{
    "status": "completed",
    "prediction": "Normal",
    "confidence": 0.92,
    "model_version": "ensemble-v4-models-tta",
    "image_url": "/media/uploads/..."
}

// Failed
{"status": "failed", "error": "Error message"}

// Not Found
{"status": "not_found"}
```

---

### Batch Classify (`batch_classify`)

**URL**: `/batch-classify/`

**Method**: POST

**Authentication**: Required

**Content-Type**: multipart/form-data

**POST Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| images | files | Yes | Multiple image files (max 10) |

**Response** (JSON):
```python
{
    "results": [
        {"image": "image1.jpg", "label": "Normal", "confidence": 0.92},
        {"image": "image2.jpg", "label": "Diseased", "confidence": 0.78},
        {"image": "image3.jpg", "error": "Image too small"}
    ]
}
```

---

## History Views

### History View (`history_view`)

**URL**: `/history/`

**Method**: GET

**Authentication**: Required

**Query Parameters**:
| Parameter | Values | Description |
|-----------|--------|-------------|
| class | "Normal", "Diseased" | Filter by prediction class |
| date | "week", "month" | Filter by time range |
| page | integer | Pagination (20 per page) |
| show_deleted | any | Include soft-deleted records |

**Context Variables**:
```python
{
    "predictions": PaginatedQuerySet,
    "total": int,          # Total records (including deleted)
    "active": int,         # Non-deleted records
    "deleted": int,        # Soft-deleted records
    "breakdown": list,     # Class distribution
    "filter_class": string,
    "date_range": string,
}
```

---

### Delete Prediction (`delete_prediction`)

**URL**: `/history/delete/<prediction_id>/`

**Method**: POST

**Authentication**: Required

**Behavior**:
- Soft-deletes prediction (sets `is_deleted=True`)
- Deletes associated image file
- Shows success message
- Redirects to history

---

### Delete All Predictions (`delete_all_predictions`)

**URL**: `/history/delete-all/`

**Method**: POST

**Authentication**: Required

**POST Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prediction_ids | string | Yes | Comma-separated IDs |

**Example**:
```
prediction_ids: "1,2,3,5,8"
```

**Behavior**: Soft-deletes specified predictions

---

### Re-analyze Prediction (`reanalyze_prediction`)

**URL**: `/history/reanalyze/<prediction_id>/`

**Method**: POST

**Authentication**: Required

**Behavior**:
- Re-runs inference on existing image
- Updates prediction record with new result
- Redirects to history

---

## Media Views

### Protected Media (`protected_media`)

**URL**: `/media/<path>/`

**Method**: GET

**Authentication**: Required

**Security Checks**:
1. Path traversal prevention
2. User ownership verification
3. MIME type validation

**Response**: FileResponse with appropriate content-type

---

## Template Mapping

| View | Template |
|------|----------|
| home_view | home.html |
| login_view | login.html |
| register_view | register.html |
| classify_retinopathy | upload.html, result.html, processing.html |
| history_view | history.html |
| batch_classify | (JSON response) |
| check_prediction_status | (JSON response) |
| protected_media | (File response) |

---

## Form Classes

### ImageUploadForm

**Fields**:
- `image`: ImageField
  - Required
  - Validators: FileExtensionValidator, validate_image_file

### LoginForm

**Fields**:
- `username`: CharField
- `password`: CharField (widget=PasswordInput)
- `remember_me`: BooleanField

### RegisterForm

**Fields**:
- `username`: CharField
- `email`: EmailField
- `password1`: CharField
- `password2`: CharField
- `organization`: CharField (optional)
- `role`: CharField (optional)
