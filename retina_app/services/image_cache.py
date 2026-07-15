"""In-memory image cache with LRU eviction."""

import hashlib
import logging
import sys
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from retina_app.constants import MAX_CACHE_MEMORY_MB, MAX_CACHE_SIZE

logger = logging.getLogger("retina_app")

IMAGE_CACHE: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
_cache_memory_bytes: int = 0


def _get_image_hash(image_path: str) -> str:
    """Compute MD5 hash of image file using 64KB chunks (avoids loading entire file)."""
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_cache_entry(cache_key: str) -> dict[str, Any] | None:
    """Retrieve an item from cache. Returns None on miss. Moves to end (most recently used)."""
    with _cache_lock:
        if cache_key in IMAGE_CACHE:
            IMAGE_CACHE.move_to_end(cache_key)
            return IMAGE_CACHE[cache_key]
    return None


def set_cache_entry(cache_key: str, result: dict[str, Any]) -> None:
    """Store a result in cache with LRU eviction and incremental memory tracking."""
    global _cache_memory_bytes
    with _cache_lock:
        # Evict LRU entries until under memory limit
        while len(IMAGE_CACHE) >= MAX_CACHE_SIZE or (
            _cache_memory_bytes > MAX_CACHE_MEMORY_MB * 1024 * 1024 and IMAGE_CACHE
        ):
            _, evicted = IMAGE_CACHE.popitem(last=False)
            _cache_memory_bytes -= _estimate_size(evicted)

        entry_size = _estimate_size(result)
        _cache_memory_bytes += entry_size
        IMAGE_CACHE[cache_key] = result


def _estimate_size(obj: Any) -> int:
    """Estimate memory size of a result dict (recursive for nested structures)."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(_estimate_size(k) + _estimate_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(_estimate_size(item) for item in obj)
    elif isinstance(obj, str):
        return sys.getsizeof(obj) + len(obj.encode("utf-8"))
    elif isinstance(obj, (int, float, bool)):
        return sys.getsizeof(obj)
    elif hasattr(obj, "nbytes"):
        return obj.nbytes
    else:
        return sys.getsizeof(obj)


def clear_image_cache() -> dict[str, int]:
    """Clear the image cache and return the number of items cleared."""
    global _cache_memory_bytes
    with _cache_lock:
        count = len(IMAGE_CACHE)
        IMAGE_CACHE.clear()
        _cache_memory_bytes = 0
        logger.info("Cleared %d items from image cache", count)
        return {"cleared_items": count, "cache_size": count}


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    with _cache_lock:
        return {
            "size": len(IMAGE_CACHE),
            "max_size": MAX_CACHE_SIZE,
            "memory_limit_mb": MAX_CACHE_MEMORY_MB,
            "memory_bytes": _cache_memory_bytes,
        }
