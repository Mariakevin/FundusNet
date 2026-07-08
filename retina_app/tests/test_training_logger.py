"""Tests for TrainingLogger."""

import json
import csv
import tempfile
import shutil
from pathlib import Path
from django.test import SimpleTestCase
from retina_app.services.training_logger import TrainingLogger


class TrainingLoggerInitTest(SimpleTestCase):
    """Test TrainingLogger initialization."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_output_dir(self):
        output = Path(self.tmp_dir) / "new_dir"
        logger = TrainingLogger(output_dir=str(output))
        self.assertTrue(output.exists())

    def test_default_run_name(self):
        logger = TrainingLogger(output_dir=self.tmp_dir)
        self.assertTrue(logger.run_name.startswith("run_"))

    def test_custom_run_name(self):
        logger = TrainingLogger(output_dir=self.tmp_dir, run_name="my_run")
        self.assertEqual(logger.run_name, "my_run")


class TrainingLoggerConfigTest(SimpleTestCase):
    """Test config logging."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.logger = TrainingLogger(output_dir=self.tmp_dir, run_name="test")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_log_config_creates_file(self):
        self.logger.log_config({"lr": 0.001, "epochs": 10})
        config_path = Path(self.tmp_dir) / "test_config.json"
        self.assertTrue(config_path.exists())

    def test_log_config_contains_values(self):
        self.logger.log_config({"lr": 0.001, "epochs": 10})
        config_path = Path(self.tmp_dir) / "test_config.json"
        with open(config_path) as f:
            data = json.load(f)
        self.assertEqual(data["lr"], 0.001)
        self.assertEqual(data["epochs"], 10)
        self.assertIn("timestamp", data)


class TrainingLoggerEpochTest(SimpleTestCase):
    """Test epoch logging."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.logger = TrainingLogger(output_dir=self.tmp_dir, run_name="test")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_log_epoch_creates_csv(self):
        self.logger.log_epoch(0, train_loss=0.5, val_acc=80.0)
        csv_path = Path(self.tmp_dir) / "test_metrics.csv"
        self.assertTrue(csv_path.exists())

    def test_log_epoch_records_data(self):
        self.logger.log_epoch(0, train_loss=0.5, val_acc=80.0)
        self.logger.log_epoch(1, train_loss=0.3, val_acc=85.0)
        self.assertEqual(len(self.logger.epochs), 2)
        self.assertEqual(self.logger.epochs[0]["epoch"], 1)
        self.assertEqual(self.logger.epochs[1]["epoch"], 2)

    def test_csv_has_correct_rows(self):
        self.logger.log_epoch(0, train_loss=0.5, val_acc=80.0)
        self.logger.log_epoch(1, train_loss=0.3, val_acc=85.0)
        csv_path = Path(self.tmp_dir) / "test_metrics.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["train_loss"]), 0.5)
        self.assertEqual(float(rows[1]["val_acc"]), 85.0)

    def test_best_val_acc_tracked(self):
        self.logger.log_epoch(0, val_acc=80.0)
        self.logger.log_epoch(1, val_acc=85.0)
        self.logger.log_epoch(2, val_acc=82.0)
        self.assertEqual(self.logger.best_val_acc, 85.0)
        self.assertEqual(self.logger.best_epoch, 2)


class TrainingLoggerFinalizeTest(SimpleTestCase):
    """Test finalize and summary."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.logger = TrainingLogger(output_dir=self.tmp_dir, run_name="test")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_finalize_creates_summary(self):
        self.logger.log_config({"lr": 0.001})
        self.logger.log_epoch(0, train_loss=0.5, val_loss=0.4, train_acc=80.0, val_acc=82.0)
        self.logger.log_epoch(1, train_loss=0.3, val_loss=0.3, train_acc=85.0, val_acc=88.0)
        summary = self.logger.finalize()
        summary_path = Path(self.tmp_dir) / "test_summary.json"
        self.assertTrue(summary_path.exists())
        self.assertIn("best_val_acc", summary)
        self.assertEqual(summary["best_val_acc"], 88.0)
        self.assertEqual(summary["best_epoch"], 2)

    def test_finalize_returns_dict(self):
        self.logger.log_epoch(0, train_loss=0.5, val_acc=80.0)
        summary = self.logger.finalize()
        self.assertIsInstance(summary, dict)
        self.assertIn("total_epochs", summary)
        self.assertIn("elapsed_seconds", summary)

    def test_get_training_curves(self):
        self.logger.log_epoch(0, train_loss=0.5, val_loss=0.4, train_acc=80.0, val_acc=82.0)
        self.logger.log_epoch(1, train_loss=0.3, val_loss=0.3, train_acc=85.0, val_acc=88.0)
        tl, vl, ta, va = self.logger.get_training_curves()
        self.assertEqual(tl, [0.5, 0.3])
        self.assertEqual(vl, [0.4, 0.3])
        self.assertEqual(ta, [80.0, 85.0])
        self.assertEqual(va, [82.0, 88.0])
