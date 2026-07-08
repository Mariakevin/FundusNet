"""
Management command to clean up old media files based on retention policy.
Deletes uploaded images older than MEDIA_RETENTION_DAYS and their associated files.
"""

from django.core.management.base import BaseCommand
from django.core.files.storage import default_storage
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
import logging

from retina_app.models import UploadedImage, PredictionRecord

logger = logging.getLogger("retina_app")


class Command(BaseCommand):
    help = "Clean up media files older than retention period"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )
        parser.add_argument(
            "--days",
            type=int,
            help="Override retention period (days)",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        days = options.get("days") or settings.MEDIA_RETENTION_DAYS

        cutoff_date = timezone.now() - timedelta(days=days)

        old_images = UploadedImage.objects.filter(uploaded_at__lt=cutoff_date)
        count = old_images.count()

        if count == 0:
            self.stdout.write("No images to clean up.")
            return

        if dry_run:
            self.stdout.write(f"DRY RUN: Would delete {count} images older than {days} days:")
            for img in old_images:
                self.stdout.write(f"  - {img.image.name} (uploaded: {img.uploaded_at})")
            return

        PredictionRecord.objects.filter(uploaded_image__in=old_images).update(is_deleted=True)

        deleted_count = 0
        for img in old_images:
            try:
                if img.image and img.image.name:
                    if default_storage.exists(img.image.name):
                        default_storage.delete(img.image.name)
                        logger.info("Deleted media file: %s", img.image.name)

                img.delete()
                deleted_count += 1

            except Exception as e:
                logger.error("Failed to delete image %s: %s", img.image.name, str(e))
                self.stderr.write(f"Error deleting {img.image.name}: {e}")

        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} images"))
        logger.info("Media cleanup completed: %d images deleted", deleted_count)
