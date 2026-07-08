"""
Production settings for RetinaAI.
Import from base and override for production.
"""

from .base import *  # noqa: F401, F403
import os

# Security - strict for production
DEBUG = os.getenv("DJANGO_DEBUG", "False").strip().lower() in {"1", "true", "yes", "on"}

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    import sys
    print("ERROR: DJANGO_SECRET_KEY environment variable is required in production!", file=sys.stderr)
    sys.exit(1)

ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "").split(",") if h.strip()]
if not ALLOWED_HOSTS:
    import sys
    print("ERROR: DJANGO_ALLOWED_HOSTS environment variable is required in production!", file=sys.stderr)
    sys.exit(1)

# Session and CSRF (secure for prod)
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_SSL_REDIRECT = os.getenv("DJANGO_SECURE_SSL_REDIRECT", "True").strip().lower() in {"1", "true", "yes", "on"}

# Security middleware headers
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

# HSTS - Force HTTPS
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Referrer Policy
REFERRER_POLICY = "strict-origin-when-cross-origin"

# Permissions Policy
PERMISSIONS_POLICY = {
    "geolocation": (),
    "microphone": (),
    "camera": (),
}

# Rate limiting - AXES for brute force protection (already installed)
# Upload rate limiting handled in views.py

# Logging - less verbose in production
LOGGING["handlers"]["console"]["level"] = "WARNING"
LOGGING["loggers"]["django"]["level"] = "WARNING"
LOGGING["loggers"]["retina_app"]["level"] = "INFO"
