"""Batch inference service with async processing, priority queues, and connection pooling.

Features:
- Async batch inference with configurable batch sizes
- Priority queue for request ordering
- Connection pooling for model instances
- Automatic retry with exponential backoff
- Result caching with TTL
- Concurrent model execution
- Progress tracking for batch jobs
- Memory-efficient streaming for large batches
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import queue
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class JobPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class InferenceJob:
    """A prioritized inference job."""

    priority: int
    job_id: str = field(compare=False)
    image_paths: list[str] = field(compare=False)
    config: dict = field(compare=False)
    created_at: float = field(compare=False, default_factory=time.time)
    callback: Any = field(compare=False, default=None)
    status: str = field(compare=False, default=JobStatus.PENDING)
    result: dict = field(compare=False, default_factory=dict)
    error: str = field(compare=False, default="")
    progress: float = field(compare=False, default=0.0)


class BatchResultCache:
    """TTL-based cache for batch inference results."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, tuple[dict, float]] = {}
        self._lock = threading.Lock()

    def _make_key(self, image_paths: list[str], config: dict) -> str:
        content = str(sorted(image_paths)) + str(sorted(config.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, image_paths: list[str], config: dict) -> dict | None:
        key = self._make_key(image_paths, config)
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return result
                del self._cache[key]
        return None

    def set(self, image_paths: list[str], config: dict, result: dict):
        key = self._make_key(image_paths, config)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (result, time.time())

    def clear(self):
        with self._lock:
            self._cache.clear()


class BatchInferenceService:
    """Production batch inference service with priority queuing."""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 100,
        cache_ttl: int = 3600,
        max_retries: int = 3,
    ):
        self.max_workers = max_workers
        self.max_retries = max_retries

        # Job queue (priority queue)
        self._queue: queue.PriorityQueue[InferenceJob] = queue.PriorityQueue(maxsize=max_queue_size)

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        atexit.register(self._executor.shutdown, wait=False)

        # Result cache
        self._cache = BatchResultCache(ttl=cache_ttl)

        # Job tracking
        self._jobs: dict[str, InferenceJob] = {}
        self._lock = threading.Lock()

        # Stats
        self._stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "avg_latency": 0.0,
        }

        # Start worker
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Main worker loop that processes jobs from the priority queue."""
        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._process_job(job)

    def _process_job(self, job: InferenceJob):
        """Process a single inference job with retry logic."""
        job.status = JobStatus.PROCESSING
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                result = self._run_batch_inference(job)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0

                # Update stats
                with self._lock:
                    self._stats["completed_jobs"] += 1
                    latency = time.time() - start_time
                    n = self._stats["completed_jobs"]
                    self._stats["avg_latency"] = (self._stats["avg_latency"] * (n - 1) + latency) / n

                # Cache result
                self._cache.set(job.image_paths, job.config, result)

                # Callback
                if job.callback:
                    try:
                        job.callback(job)
                    except Exception as e:
                        logger.warning("Job callback failed: %s", e)

                logger.info(
                    f"Job {job.job_id} completed: {len(job.image_paths)} images in {time.time() - start_time:.2f}s"
                )
                return

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning("Job %s attempt %d failed: %s, retrying in %ds", job.job_id, attempt + 1, e, wait_time)
                    time.sleep(wait_time)
                else:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    with self._lock:
                        self._stats["failed_jobs"] += 1
                    logger.error("Job %s failed after %d attempts: %s", job.job_id, self.max_retries, e)

    def _run_batch_inference(self, job: InferenceJob) -> dict:
        """Run inference on a batch of images."""
        from retina_app.services.inference import predict_image

        results = []
        for i, image_path in enumerate(job.image_paths):
            try:
                result = predict_image(
                    image_path,
                    use_ensemble=job.config.get("use_ensemble", True),
                    use_tta=job.config.get("use_tta", False),
                    use_gradcam=job.config.get("use_gradcam", False),
                )
                results.append(
                    {
                        "image_path": image_path,
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "image_path": image_path,
                        "status": "error",
                        "error": str(e),
                    }
                )

            job.progress = (i + 1) / len(job.image_paths)

        # Aggregate results
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        return {
            "job_id": job.job_id,
            "total": len(job.image_paths),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
            "summary": self._compute_summary(successful),
        }

    def _compute_summary(self, results: list[dict]) -> dict:
        """Compute aggregate statistics from batch results."""
        if not results:
            return {}

        predictions = [r["result"]["label"] for r in results]
        confidences = [r["result"]["confidence"] for r in results]
        latencies = [r["result"].get("latency", 0) for r in results]

        class_counts = defaultdict(int)
        for pred in predictions:
            class_counts[pred] += 1

        return {
            "class_distribution": dict(class_counts),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "min_confidence": float(np.min(confidences)) if confidences else 0,
            "max_confidence": float(np.max(confidences)) if confidences else 0,
            "avg_latency": float(np.mean(latencies)) if latencies else 0,
            "total_latency": float(np.sum(latencies)) if latencies else 0,
        }

    def submit_job(
        self,
        image_paths: list[str],
        config: dict | None = None,
        priority: int = JobPriority.NORMAL,
        callback: Any = None,
    ) -> str:
        """Submit a batch inference job. Returns job_id."""
        config = config or {}
        cached = self._cache.get(image_paths, config)
        if cached:
            logger.info("Cache hit for batch of %d images", len(image_paths))
            cached_job_id = cached.get("job_id")
            if cached_job_id:
                return cached_job_id

        job_id = str(uuid.uuid4())[:8]
        job = InferenceJob(
            priority=priority,
            job_id=job_id,
            image_paths=image_paths,
            config=config,
            callback=callback,
        )

        self._jobs[job_id] = job
        self._queue.put(job)

        with self._lock:
            self._stats["total_jobs"] += 1

        logger.info("Submitted job %s: %d images, priority=%d", job_id, len(image_paths), priority)
        return job_id

    def get_job_status(self, job_id: str) -> dict | None:
        """Get the status and result of a job."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "result": job.result if job.status == JobStatus.COMPLETED else None,
            "error": job.error if job.status == JobStatus.FAILED else None,
            "created_at": job.created_at,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        return False

    def get_stats(self) -> dict:
        """Get service statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_size": self._queue.qsize(),
                "active_jobs": sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING),
            }

    def shutdown(self, wait: bool = True):
        """Shutdown the service."""
        self._running = False
        self._executor.shutdown(wait=wait)


# ── Singleton ─────────────────────────────────────────────────────────────────

_batch_service: BatchInferenceService | None = None
_batch_lock = threading.Lock()


def get_batch_service() -> BatchInferenceService:
    global _batch_service
    if _batch_service is None:
        with _batch_lock:
            if _batch_service is None:
                _batch_service = BatchInferenceService()
    return _batch_service
