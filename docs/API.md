# FundusNet API Documentation

Base URL: `/api/v1/`

## Authentication

API key authentication protects prediction endpoints. Set `FUNDUSNET_API_KEYS` env var with comma-separated keys.

```bash
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "X-API-Key: your-key-here" \
  -F "image=@retinal_image.jpg"
```

If no API keys are configured, all endpoints are open access.

## Rate Limiting

- **30 requests per minute** per IP address
- Returns `429 Too Many Requests` with `Retry-After: 60` header

## Endpoints

### `GET /api/v1/`

API root with available endpoints.

**Response:**
```json
{
  "name": "FundusNet API",
  "version": "v1",
  "endpoints": {
    "predict": "/api/v1/predict/",
    "predict_batch": "/api/v1/predict/batch/",
    "jobs": "/api/v1/jobs/<job_id>/",
    "health": "/api/v1/health/",
    "registry": "/api/v1/registry/",
    "leaderboard": "/api/v1/leaderboard/",
    "experiments": "/api/v1/experiments/",
    "stats": "/api/v1/stats/"
  }
}
```

---

### `POST /api/v1/predict/`

Single image classification.

**Request:**
- `image`: File upload (multipart/form-data) — JPG, PNG, BMP, WebP, TIFF (max 10MB)
- `use_ensemble`: `"true"` or `"false"` (default: `"true"`)
- `use_tta`: `"true"` or `"false"` (default: `"false"`)
- `use_gradcam`: `"true"` or `"false"` (default: `"true"`)

**Response:**
```json
{
  "success": true,
  "result": {
    "label": "Healthy",
    "confidence": 0.95,
    "model_version": "ensemble-v3",
    "probabilities": [0.95, 0.02, 0.02, 0.01],
    "uncertainty": 0.05,
    "gradcam_url": "/media/gradcam/...",
    "is_refused": false,
    "refusal_message": "",
    "confidence_warning": null,
    "confidence_message": ""
  },
  "api_latency": 0.15
}
```

**Labels:** `Healthy`, `Cataract`, `Glaucoma`, `Retina Disease`, `Uncertain`

---

### `POST /api/v1/predict/batch/`

Batch image classification (async job).

**Request:**
```json
{
  "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "config": {},
  "priority": 2
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "a1b2c3d4",
  "status_url": "/api/v1/jobs/a1b2c3d4/",
  "total_images": 2
}
```

---

### `GET /api/v1/jobs/<job_id>/`

Check batch job status.

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "job_id": "a1b2c3d4",
    "total": 2,
    "successful": 2,
    "failed": 0,
    "results": [...],
    "summary": {
      "class_distribution": {"Healthy": 2},
      "avg_confidence": 0.92,
      "avg_latency": 0.15
    }
  },
  "error": null,
  "created_at": 1690000000.0
}
```

---

### `GET /api/v1/health/`

Model health status.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "swin": {"status": "healthy", "accuracy": 0.85, "requests": 100},
    "maxvit": {"status": "healthy", "accuracy": 0.82, "requests": 100}
  },
  "timestamp": 1690000000.0
}
```

---

### `GET /api/v1/registry/`

List registered models with performance metrics.

**Query params:** `model_name`, `limit`, `offset`

---

### `GET /api/v1/leaderboard/`

Model leaderboard ranking.

---

### `GET /api/v1/experiments/`

Training experiment tracking data.

---

### `GET /api/v1/stats/`

Batch inference service statistics.

## Error Responses

All errors return:
```json
{
  "error": "Error message"
}
```

| Status | Description |
|--------|-------------|
| 400 | Invalid input |
| 401 | Invalid or missing API key |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Server error |
