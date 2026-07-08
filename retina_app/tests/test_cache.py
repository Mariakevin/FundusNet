import os
import tempfile

from django.test import TestCase

from retina_app.constants import MAX_CACHE_SIZE
from retina_app.services.image_cache import (
    IMAGE_CACHE,
    get_cache_entry,
    set_cache_entry,
    clear_image_cache,
    get_cache_stats,
)


class ImageCacheTest(TestCase):
    def setUp(self):
        IMAGE_CACHE.clear()

    def tearDown(self):
        IMAGE_CACHE.clear()

    def test_set_and_get(self):
        set_cache_entry("key1", {"label": "Healthy", "confidence": 0.9})
        result = get_cache_entry("key1")
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "Healthy")
        self.assertEqual(result["confidence"], 0.9)

    def test_cache_miss_returns_none(self):
        result = get_cache_entry("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_roundtrip(self):
        set_cache_entry("key1", {"label": "Healthy"})
        result = get_cache_entry("key1")
        self.assertEqual(result["label"], "Healthy")

    def test_lru_eviction(self):
        for i in range(MAX_CACHE_SIZE + 5):
            set_cache_entry(f"key_{i}", {"value": i})
        self.assertLessEqual(len(IMAGE_CACHE), MAX_CACHE_SIZE)

    def test_clear_cache(self):
        set_cache_entry("key1", {"value": 1})
        set_cache_entry("key2", {"value": 2})
        result = clear_image_cache()
        self.assertEqual(result["cleared_items"], 2)
        self.assertEqual(len(IMAGE_CACHE), 0)

    def test_cache_stats(self):
        set_cache_entry("key1", {"value": 1})
        stats = get_cache_stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["max_size"], MAX_CACHE_SIZE)
        self.assertIn("memory_limit_mb", stats)
