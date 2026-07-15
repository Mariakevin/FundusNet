# FundusNet API Documentation

Base URL: `/api/v1/`

## Endpoints

### `GET /api/v1/`
API root with available endpoints.

### `POST /api/v1/predict/`
Single image classification.

**Request:**
- `image`: File upload (multipart/form-data)

**Response:**
```json
{
  "prediction": "Healthy",
  "confidence": 0.95,
  "probabilities": {"Healthy": 0.95, "Cataract": 0.02, "Glaucoma": 0.02, "Retina Disease": 0.01},
  "gradcam_url": "/media/gradcam/...",
  "preprocessing_viz_url": "/media/preprocessing_viz/...",
  "model_used": "ensemble",
  "inference_time_ms": 150
}
```

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
  "job_id": "abc123",
  "status_url": "/api/v1/jobs/abc123/",
  "total_images": 2
}
```

### `GET /api/v1/jobs/<job_id>/`
Check batch job status.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "progress": "100%",
  "results": [...]
}
```

### `GET /api/v1/health/`
Model health status.

**Response:**
```json
{
  "status": "healthy",
  "models": {"swin": "ok", "maxvit": "ok", ...},
  "uptime_seconds": 3600
}
```

### `GET /api/v1/registry/`
Model registry with performance metrics.

### `GET /api/v1/leaderboard/`
Model leaderboard ranking.

### `GET /api/v1/experiments/`
Experiment tracking data.

### `GET /api/v1/stats/`
Service statistics (cache hit rate, inference counts, etc.).

## Error Responses

All errors return:
```json
{
  "error": "Error message",
  "error_code": "ERROR_TYPE"
}
```

## Rate Limiting

- 30 requests per minute per IP
- Returns `429 Too Many Requests` when exceeded

## Authentication

Currently no authentication required (development mode).
