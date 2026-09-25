"""
Unit & Integration Test Suite — Phase 1.6 Research Evaluation, Benchmarking & Statistical Analysis
===================================================================================================
Covers:
- Evaluation schema compliance
- Isolated evaluation dataset management
- Mathematical metrics (Accuracy, Precision, Recall, F1, Specificity, Sensitivity, Balanced Accuracy)
- Agreement metrics (Cohen's Kappa, Fleiss' Kappa)
- Confusion matrices (TP, TN, FP, FN)
- Statistical summaries (Mean, Median, Std, Quartiles, IQR, Bootstrap CIs)
- Zero-denominator, NaN, Infinity, and empty dataset handling
- Error analysis and disagreement aggregation
- Side-by-side evaluation comparison and non-evaluative deltas
- Evaluation lifecycle and immutability enforcement
- Zero ground truth XML leakage and zero secret leakage
- Path traversal prevention and byte-level machine artifact immutability
- REST API endpoint verification
- Multi-format report export (JSON & text)
- Provenance and reproducibility fingerprint consistency
"""

import os
import sys
import json
import math
import hashlib
import unittest
import tempfile
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.statistical_analysis import (
    sanitize_float,
    compute_distribution_summary,
    compute_bootstrap_ci
)
from backend.error_analysis import ErrorAnalysisEngine
from backend.evaluation_engine import EvaluationEngine
from backend.evaluation_dataset_manager import EvaluationDatasetManager
from backend.evaluation_manager import EvaluationManager
from backend.evaluation_report import EvaluationReportGenerator
from backend.validate_evaluation import EvaluationValidator
from backend.experiment_comparator import ExperimentComparator
from backend.experiment_manager import ExperimentManager
from backend.dataset_snapshot_manager import DatasetSnapshotManager
from backend.api import RadiologyAPIHandler


class TestPhase16Evaluation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.eval_ds_dir = self.temp_dir / "evaluation_dataset"
        self.eval_dir = self.temp_dir / "evaluations"
        self.exp_dir = self.temp_dir / "experiments"
        self.snap_dir = self.temp_dir / "snapshots"

        self.eval_ds_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        self.ds_mgr = EvaluationDatasetManager(dataset_dir=self.eval_ds_dir)
        self.snap_mgr = DatasetSnapshotManager(snapshots_dir=self.snap_dir)
        self.exp_mgr = ExperimentManager(experiments_dir=self.exp_dir, snapshot_manager=self.snap_mgr)


        # Create base snapshot & experiment
        self.snap = self.snap_mgr.create_snapshot(name="Snap_Test", description="Test snapshot", study_ids=["CXR1122", "CXR2345"])
        self.exp = self.exp_mgr.create_experiment(
            name="Exp_Test",
            description="Test experiment",
            dataset_snapshot_id=self.snap["snapshot_id"],
            configuration={"model_name": "DenseNet-121", "qa_threshold": 0.15}
        )


        self.eval_mgr = EvaluationManager(evaluations_dir=self.eval_dir)
        self.eval_mgr.dataset_mgr = self.ds_mgr
        self.eval_mgr.exp_mgr = self.exp_mgr

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Evaluation Schema Validation
    def test_01_evaluation_schema_loads(self):
        schema_path = BASE_DIR / "docs" / "evaluation_schema.json"
        self.assertTrue(schema_path.exists())
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIn("definitions", schema)
        self.assertIn("EvaluationRun", schema["definitions"])
        self.assertIn("EvaluationDataset", schema["definitions"])

    # 2. Evaluation Dataset Creation
    def test_02_create_evaluation_dataset(self):
        ds = self.ds_mgr.create_evaluation_dataset(
            dataset_id="EVALSET_TEST_01",
            study_ids=["CXR1122", "CXR2345"],
            source_description="Unit Test Evaluation Dataset"
        )
        self.assertEqual(ds["dataset_id"], "EVALSET_TEST_01")
        self.assertEqual(ds["study_count"], 2)
        self.assertIn("manifest_hash", ds)
        self.assertTrue(self.ds_mgr.verify_dataset_integrity("EVALSET_TEST_01"))

    # 3. Dataset Fingerprinting (SHA-256)
    def test_03_dataset_fingerprint_deterministic(self):
        ds1 = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_A", study_ids=["CXR1122"])
        ds2_manifest = ds1["manifest_hash"]
        self.assertEqual(len(ds2_manifest), 64)
        # Verify integrity check matches
        self.assertTrue(self.ds_mgr.verify_dataset_integrity("EVALSET_A"))

    # 4. Evaluation Fingerprinting
    def test_04_evaluation_fingerprint_deterministic(self):
        fp1 = EvaluationManager.compute_evaluation_fingerprint("hashA", "hashB", "MethodologyA", 10)
        fp2 = EvaluationManager.compute_evaluation_fingerprint("hashA", "hashB", "MethodologyA", 10)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    # 5. Metric Calculation - Accuracy
    def test_05_accuracy_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=40, tn=40, fp=10, fn=10)
        self.assertEqual(m["accuracy"], 0.8)

    # 6. Metric Calculation - Precision
    def test_06_precision_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=30, tn=50, fp=10, fn=10)
        self.assertEqual(m["precision"], 0.75)

    # 7. Metric Calculation - Recall
    def test_07_recall_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=30, tn=50, fp=10, fn=20)
        self.assertEqual(m["recall"], 0.6)

    # 8. Metric Calculation - F1 Score
    def test_08_f1_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=30, tn=50, fp=10, fn=10)
        # precision = 30/40 = 0.75, recall = 30/40 = 0.75 -> f1 = 0.75
        self.assertEqual(m["f1"], 0.75)

    # 9. Metric Calculation - Specificity
    def test_09_specificity_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=30, tn=40, fp=10, fn=10)
        # specificity = 40/(40+10) = 0.8
        self.assertEqual(m["specificity"], 0.8)

    # 10. Metric Calculation - Sensitivity
    def test_10_sensitivity_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=25, tn=50, fp=10, fn=25)
        # sensitivity = recall = 25/50 = 0.5
        self.assertEqual(m["sensitivity"], 0.5)

    # 11. Metric Calculation - Balanced Accuracy
    def test_11_balanced_accuracy_calculation(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=40, tn=30, fp=10, fn=10)
        # sensitivity = 40/50 = 0.8, specificity = 30/40 = 0.75 -> bal_acc = 0.775
        self.assertEqual(m["balanced_accuracy"], 0.775)

    # 12. Agreement Metric - Cohen's Kappa
    def test_12_cohen_kappa_calculation(self):
        r1 = [1, 1, 0, 0, 1, 0, 1, 1]
        r2 = [1, 1, 0, 0, 1, 0, 0, 1]
        k = EvaluationEngine.compute_cohen_kappa(r1, r2)
        self.assertIsNotNone(k)
        self.assertGreaterEqual(k, -1.0)
        self.assertLessEqual(k, 1.0)

    # 13. Agreement Metric - Fleiss' Kappa
    def test_13_fleiss_kappa_calculation(self):
        # 4 subjects, 3 raters, 2 categories [pos, neg]
        matrix = [
            [3, 0],
            [3, 0],
            [0, 3],
            [2, 1]
        ]
        k = EvaluationEngine.compute_fleiss_kappa(matrix)
        self.assertIsNotNone(k)
        self.assertGreaterEqual(k, -1.0)
        self.assertLessEqual(k, 1.0)

    # 14. Confusion Matrix Construction
    def test_14_confusion_matrix_structure(self):
        cm = ErrorAnalysisEngine.build_confusion_matrix(tp=12, tn=34, fp=5, fn=2)
        self.assertEqual(cm["tp"], 12)
        self.assertEqual(cm["tn"], 34)
        self.assertEqual(cm["fp"], 5)
        self.assertEqual(cm["fn"], 2)

    # 15. Statistical Summaries (Mean, Median, Std, Quartiles)
    def test_15_statistical_distribution_summary(self):
        vals = [0.80, 0.82, 0.85, 0.90, 0.92]
        d = compute_distribution_summary(vals)
        self.assertAlmostEqual(d["mean"], 0.858, places=2)
        self.assertEqual(d["median"], 0.85)
        self.assertEqual(d["min"], 0.80)
        self.assertEqual(d["max"], 0.92)
        self.assertEqual(d["count"], 5)

    # 16. Empty Dataset Handling
    def test_16_empty_dataset_handling(self):
        d = compute_distribution_summary([])
        self.assertEqual(d["count"], 0)
        self.assertEqual(d["mean"], 0.0)
        ci = compute_bootstrap_ci([])
        self.assertEqual(ci["lower_95"], 0.0)

    # 17. Single Observation Dataset Handling
    def test_17_single_observation_handling(self):
        d = compute_distribution_summary([0.75])
        self.assertEqual(d["count"], 1)
        self.assertEqual(d["mean"], 0.75)
        self.assertEqual(d["std"], 0.0)

    # 18. Zero Denominator Handling
    def test_18_zero_denominator_handling(self):
        m = ErrorAnalysisEngine.calculate_classification_metrics(tp=0, tn=0, fp=0, fn=0)
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["f1"], 0.0)

    # 19. NaN Rejection
    def test_19_nan_rejection(self):
        val = sanitize_float(float("nan"), default=0.0)
        self.assertEqual(val, 0.0)
        self.assertFalse(math.isnan(val))

    # 20. Infinity Rejection
    def test_20_infinity_rejection(self):
        val = sanitize_float(float("inf"), default=1.0)
        self.assertEqual(val, 1.0)
        self.assertFalse(math.isinf(val))

    # 21. Error Analysis Breakdown
    def test_21_error_analysis_breakdown(self):
        eval_findings = [
            {"finding_type": "cardiomegaly", "sample_count": 10, "confusion_matrix": {"tp": 5, "tn": 3, "fp": 1, "fn": 1}},
            {"finding_type": "pleural_effusion", "sample_count": 10, "confusion_matrix": {"tp": 4, "tn": 4, "fp": 1, "fn": 1}}
        ]
        res = ErrorAnalysisEngine.analyze_finding_disagreements(eval_findings)
        self.assertEqual(res["total_disagreements"], 4)
        self.assertEqual(res["per_finding_disagreements"]["cardiomegaly"], 2)

    # 22. Experiment Comparison Extension
    def test_22_experiment_comparator_evaluations(self):
        eval_a = {
            "evaluation_id": "EVAL_A", "experiment_id": "EXP_1",
            "aggregate_metrics": {"accuracy": {"metric_name": "Accuracy", "metric_value": 0.80, "sample_size": 20}}
        }
        eval_b = {
            "evaluation_id": "EVAL_B", "experiment_id": "EXP_2",
            "aggregate_metrics": {"accuracy": {"metric_name": "Accuracy", "metric_value": 0.85, "sample_size": 20}}
        }
        comp = ExperimentComparator().compare_evaluations(eval_a, eval_b)
        self.assertEqual(comp["comparison_id"], "COMP_EVAL_A_EVAL_B")
        self.assertEqual(len(comp["metric_comparisons"]), 1)
        self.assertAlmostEqual(comp["metric_comparisons"][0]["delta"], 0.05, places=2)

    # 23. Non-Evaluative Delta Terminology
    def test_23_non_evaluative_comparison_wording(self):
        eval_a = {"evaluation_id": "EVAL_A", "aggregate_metrics": {}}
        eval_b = {"evaluation_id": "EVAL_B", "aggregate_metrics": {}}
        comp = ExperimentComparator().compare_evaluations(eval_a, eval_b)
        serialized = json.dumps(comp).lower()
        forbidden = ["best model", "winner", "superior", "recommended"]
        for word in forbidden:
            self.assertNotIn(word, serialized)

    # 24. Evaluation Run Lifecycle
    def test_24_evaluation_lifecycle(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_RUN", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_RUN",
            evaluation_id="EVAL_LIFECYCLE_01"
        )
        self.assertEqual(ev["status"], "CREATED")

        completed = self.eval_mgr.run_evaluation("EVAL_LIFECYCLE_01")
        self.assertEqual(completed["status"], "COMPLETED")
        self.assertIsNotNone(completed["completed_at"])

        validated = self.eval_mgr.validate_evaluation("EVAL_LIFECYCLE_01")
        self.assertEqual(validated["status"], "VALIDATED")

        archived = self.eval_mgr.archive_evaluation("EVAL_LIFECYCLE_01")
        self.assertEqual(archived["status"], "ARCHIVED")

    # 25. Completed Evaluation Immutability
    def test_25_completed_evaluation_immutability(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_IMMUTABLE", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_IMMUTABLE",
            evaluation_id="EVAL_IMMUTABLE_01"
        )
        self.eval_mgr.run_evaluation("EVAL_IMMUTABLE_01")
        # Trying to rerun completed evaluation must raise ValueError
        with self.assertRaises(ValueError):
            self.eval_mgr.run_evaluation("EVAL_IMMUTABLE_01")

    # 26. Archived Evaluation Immutability
    def test_26_archived_evaluation_immutability(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_ARCH", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_ARCH",
            evaluation_id="EVAL_ARCH_01"
        )
        self.eval_mgr.archive_evaluation("EVAL_ARCH_01")
        with self.assertRaises(ValueError):
            self.eval_mgr.run_evaluation("EVAL_ARCH_01")

    # 27. Ground Truth XML Leakage Prevention
    def test_27_ground_truth_leakage_protection(self):
        eval_doc = {
            "evaluation_id": "EVAL_CLEAN",
            "aggregate_metrics": {"accuracy": {"metric_value": 0.85}}
        }
        errs = EvaluationValidator.check_zero_ground_truth_leakage(eval_doc)
        self.assertEqual(len(errs), 0)

        leaky_doc = {"evaluation_id": "EVAL_LEAK", "notes": "Found <eFind>cardiomegaly</eFind>"}
        errs_leak = EvaluationValidator.check_zero_ground_truth_leakage(leaky_doc)
        self.assertGreater(len(errs_leak), 0)

    # 28. Secret Leakage Prevention
    def test_28_secret_leakage_protection(self):
        clean_doc = {"evaluation_id": "EVAL_SAFE", "method": "Standard"}
        self.assertEqual(len(EvaluationValidator.check_zero_secrets(clean_doc)), 0)

        dirty_doc = {"evaluation_id": "EVAL_KEY", "key": "sk-1234567890abcdef"}
        self.assertGreater(len(EvaluationValidator.check_zero_secrets(dirty_doc)), 0)

    # 29. Path Traversal Prevention
    def test_29_path_traversal_prevention(self):
        with self.assertRaises(ValueError):
            self.eval_mgr._validate_safe_id("../../bad_id")

        with self.assertRaises(ValueError):
            self.ds_mgr._validate_safe_id("eval/traversal")

    # 30. Machine Artifact Byte Immutability
    def test_30_machine_artifact_immutability(self):
        iu_dir = BASE_DIR / "data" / "iu_xray"
        hashes_before = EvaluationValidator.calculate_dir_sha256(iu_dir)

        # Run an evaluation
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_HASH", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_HASH",
            evaluation_id="EVAL_HASH_01"
        )
        self.eval_mgr.run_evaluation("EVAL_HASH_01")

        hashes_after = EvaluationValidator.calculate_dir_sha256(iu_dir)
        self.assertEqual(hashes_before, hashes_after)

    # 31. Evaluation Report JSON Export
    def test_31_report_json_export(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_REP", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_REP",
            evaluation_id="EVAL_REP_01"
        )
        self.eval_mgr.run_evaluation("EVAL_REP_01")
        doc = self.eval_mgr.get_evaluation("EVAL_REP_01")
        report = EvaluationReportGenerator.generate_json_report(doc)
        self.assertEqual(report["report_id"], "REP_EVAL_REP_01")
        self.assertIn("observed_metrics", report)
        self.assertIn("disclaimer", report)

    # 32. Evaluation Report Text Export
    def test_32_report_text_export(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_TXT", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_TXT",
            evaluation_id="EVAL_TXT_01"
        )
        self.eval_mgr.run_evaluation("EVAL_TXT_01")
        doc = self.eval_mgr.get_evaluation("EVAL_TXT_01")
        txt = EvaluationReportGenerator.generate_text_report(doc)
        self.assertIn("EXPLAINABLE RADIOLOGY RESEARCH PROTOTYPE", txt)
        self.assertIn("MANDATORY RESEARCH DISCLAIMER", txt)

    # 33. Evaluation Provenance Completeness
    def test_33_provenance_audit_trail(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_PROV", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_PROV",
            evaluation_id="EVAL_PROV_01"
        )
        self.eval_mgr.run_evaluation("EVAL_PROV_01")
        doc = self.eval_mgr.get_evaluation("EVAL_PROV_01")
        prov = doc.get("provenance", {})
        self.assertIn("status_history", prov)
        self.assertGreaterEqual(len(prov["status_history"]), 2)

    # 34. Invariant Validator Full Check
    def test_34_evaluation_validator_full_check(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_VAL", study_ids=["CXR1122"])
        ev = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_VAL",
            evaluation_id="EVAL_VAL_01"
        )
        self.eval_mgr.run_evaluation("EVAL_VAL_01")
        doc = self.eval_mgr.get_evaluation("EVAL_VAL_01")
        res = EvaluationValidator.validate_evaluation_record(doc)
        self.assertTrue(res["valid"])
        self.assertEqual(res["error_count"], 0)

    # 35. Reproducibility Fingerprint Consistency
    def test_35_reproducibility_consistency(self):
        ds = self.ds_mgr.create_evaluation_dataset(dataset_id="EVALSET_REPRO", study_ids=["CXR1122"])
        ev1 = self.eval_mgr.create_evaluation(
            experiment_id=self.exp["experiment_id"],
            evaluation_dataset_id="EVALSET_REPRO",
            evaluation_id="EVAL_REPRO_01"
        )
        fp1 = ev1["evaluation_fingerprint"]

        # Second creation with same parameters
        fp2 = EvaluationManager.compute_evaluation_fingerprint(
            dataset_fingerprint=ev1["dataset_fingerprint"],
            configuration_fingerprint=ev1["configuration_fingerprint"],
            methodology="Standard Research Evaluation Benchmark",
            study_count=ev1["study_count"]
        )
        self.assertEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
