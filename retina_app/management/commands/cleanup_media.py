"""
Management command to clean up old media files based on retention policy.
Deletes uploaded images older than MEDIA_RETENTION_DAYS and their associated files.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
import os
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

        # Find old images
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

        # Delete images and files
        deleted_count = 0
        for img in old_images:
            try:
                # Soft-delete prediction records first
                PredictionRecord.objects.filter(uploaded_image=img).update(is_deleted=True)

                # Delete file from storage
                if img.image and os.path.exists(img.image.path):
                    os.remove(img.image.path)
                    logger.info("Deleted media file: %s", img.image.name)

                # Delete the database record (will SET_NULL on PredictionRecord.uploaded_image)
                img.delete()
                deleted_count += 1

            except Exception as e:
                logger.error("Failed to delete image %s: %s", img.image.name, str(e))
                self.stderr.write(f"Error deleting {img.image.name}: {e}")

        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} images"))
        logger.info("Media cleanup completed: %d images deleted", deleted_count)
