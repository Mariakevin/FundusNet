"""
Development settings for RetinaAI.
Import from base and override for development.
"""

from .base import *  # noqa: F401, F403
from pathlib import Path
import os

# Security - relaxed for development
DEBUG = os.getenv("DJANGO_DEBUG", "True").strip().lower() in {"1", "true", "yes", "on"}

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "django-insecure-dev-key-change-me")

ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]

# Session and CSRF (relaxed for dev)
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = False

# Axes - disable lockout in dev for testing
AXES_LOCK_OUT_AT_FAILURE = False
AXES_FAILURE_LIMIT = 10

# Logging - more verbose in dev
LOGGING["loggers"]["django"]["level"] = "DEBUG"
LOGGING["loggers"]["retina_app"]["level"] = "DEBUG"
