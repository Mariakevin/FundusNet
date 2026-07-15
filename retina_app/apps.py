import logging

from django.apps import AppConfig

logger = logging.getLogger("retina_app")


class RetinaAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "retina_app"
    verbose_name = "FundusNet"

    def ready(self):
        """Start background model preloading when Django starts."""
        try:
            from retina_app.services.model_manager import preload_models_background
            preload_models_background()
        except Exception as exc:
            logger.warning("Could not start model preloader: %s", exc)
