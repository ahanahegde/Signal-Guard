"""
SignalGuard Baseline Classifier Test Suite.

Tests:
1. DatasetLoader manifest reading, writing, and identity-leakage prevention.
2. Synthetic fixture generation with explicit synthetic tags.
3. Feature extraction (spatial, frequency, temporal) and error handling on bad files.
4. Classifier fitting, probability calibration, serialization, and deserialization.
5. Standalone inference interface (predict_video).
6. Comprehensive evaluation metrics computation, confusion matrix, and edge cases.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline_classifier import (
    FEATURE_NAMES,
    BaselineClassifier,
    extract_frame_forensic_features,
    extract_video_features,
    predict_video,
)
from classifier_evaluator import (
    compute_metrics_from_arrays,
    evaluate_records,
)
from dataset_loader import (
    LABEL_GENUINE,
    LABEL_MANIPULATED,
    DatasetLoader,
    VideoRecord,
)


class TestDatasetLoader(unittest.TestCase):
    """Test dataset loading, manifest roundtrip, and identity leakage prevention."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sg_test_dataset_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manifest_roundtrip_csv_and_json(self):
        records = [
            VideoRecord(
                video_path="/data/real_001.mp4",
                label=LABEL_GENUINE,
                label_name="GENUINE",
                subject_id="sub_A",
                condition="c23",
                split="train",
                is_synthetic=False,
            ),
            VideoRecord(
                video_path="/data/fake_001.mp4",
                label=LABEL_MANIPULATED,
                label_name="MANIPULATED",
                subject_id="sub_B",
                condition="Deepfakes",
                split="test",
                is_synthetic=False,
            ),
        ]

        # CSV test
        csv_path = self.temp_dir / "manifest.csv"
        DatasetLoader.save_manifest(records, csv_path)
        loaded_csv = DatasetLoader.load_from_manifest(csv_path)
        self.assertEqual(len(loaded_csv), 2)
        self.assertEqual(loaded_csv[0].subject_id, "sub_A")
        self.assertEqual(loaded_csv[1].label, LABEL_MANIPULATED)

        # JSON test
        json_path = self.temp_dir / "manifest.json"
        DatasetLoader.save_manifest(records, json_path)
        loaded_json = DatasetLoader.load_from_manifest(json_path)
        self.assertEqual(len(loaded_json), 2)
        self.assertEqual(loaded_json[1].condition, "Deepfakes")

    def test_leakage_prevention_split_by_subject(self):
        # Create records across 10 distinct subjects
        records = []
        for i in range(10):
            sub = f"subject_{i:02d}"
            # 2 clips per subject
            records.append(VideoRecord(f"vid_{i}_1.mp4", LABEL_GENUINE, "GENUINE", sub))
            records.append(VideoRecord(f"vid_{i}_2.mp4", LABEL_MANIPULATED, "MANIPULATED", sub))

        train_r, val_r, test_r = DatasetLoader.split_by_subject(
            records, val_ratio=0.20, test_ratio=0.20, random_seed=42
        )

        train_subs = set(r.subject_id for r in train_r)
        val_subs = set(r.subject_id for r in val_r)
        test_subs = set(r.subject_id for r in test_r)

        # Strictly zero identity overlap
        self.assertEqual(len(train_subs.intersection(test_subs)), 0, "Identity leakage detected between train and test!")
        self.assertEqual(len(train_subs.intersection(val_subs)), 0, "Identity leakage detected between train and val!")
        self.assertEqual(len(val_subs.intersection(test_subs)), 0, "Identity leakage detected between val and test!")

        # All records partitioned
        self.assertEqual(len(train_r) + len(val_r) + len(test_r), len(records))


class TestFeatureExtractionAndClassifier(unittest.TestCase):
    """Test feature extraction, model fitting, calibration, and prediction."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = Path(tempfile.mkdtemp(prefix="sg_test_fixture_"))
        # Generate a small synthetic fixture: 4 subjects x 4 clips = 16 clips
        cls.records = DatasetLoader.generate_synthetic_fixture(
            cls.fixture_dir,
            num_subjects=4,
            clips_per_subject=4,
            fps=12.0,
            duration_seconds=1.5,
            width=160,
            height=120,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixture_dir, ignore_errors=True)

    def test_single_frame_feature_extraction(self):
        dummy_frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        feat = extract_frame_forensic_features(dummy_frame)
        self.assertIn("high_freq_ratio", feat)
        self.assertIn("azimuthal_spectral_variance", feat)
        self.assertIn("laplacian_variance", feat)
        self.assertTrue(np.isfinite(feat["high_freq_ratio"]))

    def test_video_feature_extraction(self):
        sample_video = self.records[0].video_path
        vec, feat_dict, err = extract_video_features(sample_video)
        self.assertIsNone(err)
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_error_handling_unreadable_video(self):
        vec, feat_dict, err = extract_video_features("non_existent_file.mp4")
        self.assertIsNone(vec)
        self.assertIsNotNone(err)

    def test_classifier_fit_calibrate_and_predict(self):
        train_r, val_r, test_r = DatasetLoader.split_by_subject(
            self.records, val_ratio=0.25, test_ratio=0.25, random_seed=42
        )

        classifier = BaselineClassifier(c_regularization=1.0, decision_threshold=0.5)
        meta = classifier.fit(train_r, val_records=val_r)

        self.assertIn("n_samples", meta)
        self.assertGreater(meta["n_samples"], 0)

        # Test single prediction
        test_video = test_r[0].video_path
        pred_res = classifier.predict_single(test_video)

        self.assertEqual(pred_res["status"], "SUCCESS")
        self.assertIn(pred_res["prediction"], ["GENUINE", "MANIPULATED"])
        self.assertGreaterEqual(pred_res["probability_manipulated"], 0.0)
        self.assertLessEqual(pred_res["probability_manipulated"], 1.0)
        self.assertAlmostEqual(
            pred_res["probability_manipulated"] + pred_res["probability_genuine"],
            1.0,
            places=4,
        )

        # Test save and reload
        save_model_path = self.fixture_dir / "test_model.joblib"
        save_config_path = self.fixture_dir / "test_config.json"
        classifier.save(save_model_path, save_config_path)

        loaded_clf = BaselineClassifier.load(save_model_path)
        pred_reloaded = loaded_clf.predict_single(test_video)
        self.assertEqual(pred_res["prediction"], pred_reloaded["prediction"])
        self.assertAlmostEqual(
            pred_res["probability_manipulated"],
            pred_reloaded["probability_manipulated"],
            places=4,
        )

        # Test standalone predict_video function
        func_res = predict_video(test_video, model_path=save_model_path)
        self.assertEqual(func_res["status"], "SUCCESS")
        self.assertEqual(func_res["prediction"], pred_res["prediction"])


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metric calculations, confusion matrix, and edge cases."""

    def test_perfect_metrics(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]

        res = compute_metrics_from_arrays(y_true, y_pred, y_prob)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["specificity"], 1.0)
        self.assertEqual(res["false_positive_rate"], 0.0)
        self.assertEqual(res["false_negative_rate"], 0.0)
        self.assertEqual(res["balanced_accuracy"], 1.0)
        self.assertEqual(res["roc_auc"], 1.0)

    def test_confusion_matrix_accounting(self):
        # 2 TN, 1 FP, 1 FN, 2 TP
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 0, 1, 1]

        res = compute_metrics_from_arrays(y_true, y_pred)
        cm = res["confusion_matrix"]
        self.assertEqual(cm["true_negatives"], 2)
        self.assertEqual(cm["false_positives"], 1)
        self.assertEqual(cm["false_negatives"], 1)
        self.assertEqual(cm["true_positives"], 2)

        # FPR = FP / (FP + TN) = 1 / 3 = 0.3333
        self.assertAlmostEqual(res["false_positive_rate"], 1.0 / 3.0, places=4)
        # FNR = FN / (FN + TP) = 1 / 3 = 0.3333
        self.assertAlmostEqual(res["false_negative_rate"], 1.0 / 3.0, places=4)


    def test_evaluate_records_on_dataset(self):
        # Generate small dataset fixture
        fixture_dir = Path(tempfile.mkdtemp(prefix="sg_eval_test_"))
        try:
            records = DatasetLoader.generate_synthetic_fixture(
                fixture_dir, num_subjects=4, clips_per_subject=2, fps=10.0, duration_seconds=1.0, width=120, height=90
            )
            classifier = BaselineClassifier()
            classifier.fit(records)
            metrics = evaluate_records(classifier, records, split_name="test_fixture")
            self.assertIn("balanced_accuracy", metrics)
            self.assertIn("f1_score", metrics)
            self.assertEqual(metrics["total_samples"], len(records))
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
