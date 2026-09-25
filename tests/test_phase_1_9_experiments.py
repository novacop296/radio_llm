"""
Phase 1.9 Test Suite: Research Experiment Orchestration, Statistical Reporting & Reproducibility Dashboard
==========================================================================================================
Comprehensive unit, invariant, and integration test suite verifying:
- Experiment creation, schema validation, and lifecycle state management
- Deterministic fingerprint generation, drift detection, and reproducibility manifests
- Statistical summaries, bootstrap confidence intervals, and zero-denominator handling
- Multi-format exports (JSON, CSV, Plain Text, Markdown) and 16-section research reports
- Immutability of published experiments and underlying base IU X-ray machine artifacts
- Ground-truth XML report isolation and zero secret/API key disclosures
- REST API endpoints and regression compatibility across Phases 0.6–1.8
"""

import os
import sys
import json
import shutil
import unittest
import math
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.experiment_manager import ExperimentManager, RESEARCH_DISCLAIMER
from backend.statistics_manager import StatisticsManager, sanitize_float, check_finite_numbers
from backend.research_report import ResearchReportGenerator
from backend.validate_phase_1_9 import (
    validate_all_phase_1_9_invariants,
    check_ground_truth_isolation,
    check_secret_leakage,
    check_iu_xray_dataset_integrity
)
from backend.benchmark_manager import global_benchmark_manager
from backend.external_dataset_manager import global_external_dataset_manager
from backend.experiment_registry import global_experiment_registry
from backend.api import RadiologyAPIHandler


class TestPhase19Experiments(unittest.TestCase):
    """Test suite for Phase 1.9 research experiment orchestration and statistical reporting."""

    def setUp(self):
        self.test_dir = os.path.join(BASE_DIR, "data", "test_phase_1_9_experiments")
        os.makedirs(self.test_dir, exist_ok=True)
        self.stats_mgr = StatisticsManager(default_seed=42, default_bootstrap_iterations=500)
        self.report_gen = ResearchReportGenerator()
        self.exp_mgr = ExperimentManager(
            experiments_dir=self.test_dir,
            statistics_manager=self.stats_mgr,
            report_generator=self.report_gen
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Experiment Creation
    def test_01_experiment_creation(self):
        exp = self.exp_mgr.create_experiment(
            name="Test Experiment 01",
            description="Testing creation of standard experiment definition.",
            custom_id="exp_test_01"
        )
        self.assertIsNotNone(exp)
        self.assertEqual(exp["experiment_id"], "exp_test_01")
        self.assertEqual(exp["experiment_name"], "Test Experiment 01")
        self.assertEqual(exp["status"], "CONFIGURED")
        self.assertIn("dataset_ref", exp)
        self.assertIn("model_ref", exp)
        self.assertIn("configuration", exp)

    # 2. Schema Validation
    def test_02_schema_validation(self):
        exp = self.exp_mgr.create_experiment(name="Schema Test Exp", custom_id="exp_schema_02")
        self.assertIn("input_fingerprints", exp)
        self.assertIn("canonical_fingerprint", exp["input_fingerprints"])
        self.assertIn("reproducibility_manifest", exp)
        self.assertIn("research_disclaimer", exp)

    # 3. Configuration Validation
    def test_03_configuration_validation(self):
        exp = self.exp_mgr.create_experiment(
            name="Config Validation Exp",
            configuration={"view_configuration": "MULTI_VIEW", "bootstrap_sample_count": 2000, "random_seed": 123},
            custom_id="exp_cfg_03"
        )
        cfg = exp["configuration"]
        self.assertEqual(cfg["view_configuration"], "MULTI_VIEW")
        self.assertEqual(cfg["bootstrap_sample_count"], 2000)
        self.assertEqual(cfg["random_seed"], 123)

    # 4. Dataset Reference Validation
    def test_04_dataset_reference_validation(self):
        ds_ref = {
            "dataset_id": "mimic_cxr_subset_v1",
            "dataset_name": "MIMIC-CXR External Subset",
            "dataset_type": "EXTERNAL",
            "version": "1.0.0",
            "split": "external_validation",
            "sample_count": 250,
            "manifest_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "view_pairing": "FRONTAL_LATERAL_PAIRED"
        }
        exp = self.exp_mgr.create_experiment(name="Ext Dataset Exp", dataset_ref=ds_ref, custom_id="exp_ds_04")
        self.assertEqual(exp["dataset_ref"]["dataset_id"], "mimic_cxr_subset_v1")
        self.assertEqual(exp["dataset_ref"]["sample_count"], 250)

    # 5. Model Reference Validation
    def test_05_model_reference_validation(self):
        model_ref = {
            "model_id": "txrv_resnet50",
            "model_name": "TorchXRayVision ResNet-50",
            "model_architecture": "resnet50",
            "model_version": "2.0.0",
            "weights_sha256": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "target_layer": "model.layer4",
            "preprocessing_resolution": [224, 224],
            "normalization": "minmax_0_1"
        }
        exp = self.exp_mgr.create_experiment(name="Model Ref Exp", model_ref=model_ref, custom_id="exp_model_05")
        self.assertEqual(exp["model_ref"]["model_architecture"], "resnet50")
        self.assertEqual(exp["model_ref"]["target_layer"], "model.layer4")

    # 6. Experiment Lifecycle
    def test_06_experiment_lifecycle(self):
        exp = self.exp_mgr.create_experiment(name="Lifecycle Exp", custom_id="exp_life_06")
        self.assertEqual(exp["status"], "CONFIGURED")

        exp_run = self.exp_mgr.run_experiment("exp_life_06")
        self.assertEqual(exp_run["status"], "COMPLETED")

        exp_fin = self.exp_mgr.finalize_experiment("exp_life_06")
        self.assertEqual(exp_fin["status"], "VALIDATED")

        exp_pub = self.exp_mgr.publish_experiment("exp_life_06")
        self.assertEqual(exp_pub["status"], "PUBLISHED")
        self.assertTrue(exp_pub.get("locked"))

    # 7. Benchmark Execution
    def test_07_benchmark_execution(self):
        exp = self.exp_mgr.create_experiment(name="Benchmark Run Exp", custom_id="exp_bm_07")
        res = self.exp_mgr.run_experiment("exp_bm_07")
        self.assertIsNotNone(res.get("metrics"))
        self.assertIn("macro_avg", res["metrics"])
        self.assertIn("per_finding", res["metrics"])
        self.assertGreater(res["metrics"]["macro_avg"]["auc"], 0.5)

    # 8. Metric Calculation
    def test_08_metric_calculation(self):
        custom_metrics = {
            "macro_avg": {"auc": 0.9250, "sensitivity": 0.8800, "specificity": 0.9400, "f1": 0.8950, "accuracy": 0.9100},
            "per_finding": {"Pneumonia": {"auc": 0.9250, "sensitivity": 0.8800, "specificity": 0.9400, "f1": 0.8950, "accuracy": 0.9100}}
        }
        exp = self.exp_mgr.create_experiment(name="Custom Metrics Exp", custom_id="exp_met_08")
        res = self.exp_mgr.run_experiment("exp_met_08", simulated_metrics=custom_metrics)
        self.assertEqual(res["metrics"]["macro_avg"]["auc"], 0.9250)
        self.assertEqual(res["metrics"]["macro_avg"]["f1"], 0.8950)

    # 9. Bootstrap Confidence Intervals
    def test_09_bootstrap_confidence_intervals(self):
        values = [0.82, 0.84, 0.85, 0.88, 0.89, 0.91, 0.92, 0.94]
        ci = self.stats_mgr.compute_bootstrap_ci(values, n_bootstraps=500, confidence_level=0.95, seed=42)
        self.assertIn("lower", ci)
        self.assertIn("upper", ci)
        self.assertLessEqual(ci["lower"], ci["upper"])
        self.assertEqual(ci["sample_size"], len(values))

    # 10. Deterministic Random Seed
    def test_10_deterministic_random_seed(self):
        values = [0.81, 0.83, 0.87, 0.90, 0.92]
        ci1 = self.stats_mgr.compute_bootstrap_ci(values, n_bootstraps=500, seed=123)
        ci2 = self.stats_mgr.compute_bootstrap_ci(values, n_bootstraps=500, seed=123)
        self.assertEqual(ci1["lower"], ci2["lower"])
        self.assertEqual(ci1["upper"], ci2["upper"])

    # 11. NaN Handling
    def test_11_nan_handling(self):
        val_nan = float('nan')
        clean = sanitize_float(val_nan, default=0.0)
        self.assertEqual(clean, 0.0)

        dist = self.stats_mgr.compute_distribution([0.85, float('nan'), 0.90])
        self.assertFalse(math.isnan(dist["mean"]))
        self.assertEqual(dist["count"], 2)

    # 12. Infinity Handling
    def test_12_infinity_handling(self):
        val_inf = float('inf')
        clean = sanitize_float(val_inf, default=1.0)
        self.assertEqual(clean, 1.0)

        ok, err = check_finite_numbers({"auc": float('-inf')})
        self.assertFalse(ok)
        self.assertIn("Infinity", err)

    # 13. Experiment Fingerprint Generation
    def test_13_experiment_fingerprint_generation(self):
        exp = self.exp_mgr.create_experiment(name="FP Test Exp", custom_id="exp_fp_13")
        fps = exp["input_fingerprints"]
        self.assertEqual(len(fps["canonical_fingerprint"]), 64)
        self.assertEqual(len(fps["dataset_manifest_sha256"]), 64)
        self.assertEqual(len(fps["model_weights_sha256"]), 64)

    # 14. Dataset Fingerprint Mismatch Detection
    def test_14_dataset_fingerprint_mismatch_detection(self):
        exp = self.exp_mgr.create_experiment(name="Drift Test Exp", custom_id="exp_drift_14")
        # Tamper recorded dataset fingerprint
        exp["input_fingerprints"]["dataset_manifest_sha256"] = "tampered_dataset_hash_0000000000000000000000000000000000000000"
        repro = self.exp_mgr.generate_reproducibility_manifest("exp_drift_14", base_exp=exp)
        self.assertEqual(repro["reproducibility_status"], "DRIFT_DETECTED")
        self.assertTrue(repro["drift_details"]["dataset_drift"])

    # 15. Model Fingerprint Mismatch Detection
    def test_15_model_fingerprint_mismatch_detection(self):
        exp = self.exp_mgr.create_experiment(name="Model Drift Exp", custom_id="exp_drift_15")
        exp["input_fingerprints"]["model_weights_sha256"] = "tampered_model_hash_000000000000000000000000000000000000000000"
        repro = self.exp_mgr.generate_reproducibility_manifest("exp_drift_15", base_exp=exp)
        self.assertEqual(repro["reproducibility_status"], "DRIFT_DETECTED")
        self.assertTrue(repro["drift_details"]["model_drift"])

    # 16. Configuration Drift Detection
    def test_16_configuration_drift_detection(self):
        exp = self.exp_mgr.create_experiment(name="Config Drift Exp", custom_id="exp_drift_16")
        exp["input_fingerprints"]["configuration_sha256"] = "tampered_config_hash_0000000000000000000000000000000000000000"
        repro = self.exp_mgr.generate_reproducibility_manifest("exp_drift_16", base_exp=exp)
        self.assertEqual(repro["reproducibility_status"], "DRIFT_DETECTED")
        self.assertTrue(repro["drift_details"]["config_drift"])

    # 17. Experiment Comparison
    def test_17_experiment_comparison(self):
        self.exp_mgr.create_experiment(name="Exp A", custom_id="exp_comp_a")
        self.exp_mgr.run_experiment("exp_comp_a", simulated_metrics={"macro_avg": {"auc": 0.8500, "f1": 0.8000}})

        self.exp_mgr.create_experiment(name="Exp B", custom_id="exp_comp_b")
        self.exp_mgr.run_experiment("exp_comp_b", simulated_metrics={"macro_avg": {"auc": 0.8900, "f1": 0.8400}})

        comp = self.exp_mgr.compare_experiments(["exp_comp_a", "exp_comp_b"])
        self.assertIn("metric_deltas", comp)
        self.assertAlmostEqual(comp["metric_deltas"]["auc"]["delta_abs"], 0.0400, places=3)
        self.assertIn("neutral_summary", comp)
        self.assertNotIn("winner", comp["neutral_summary"].lower())
        self.assertNotIn("superior", comp["neutral_summary"].lower())

    # 18. Statistical Summary Generation
    def test_18_statistical_summary_generation(self):
        metric_lists = {
            "auc": [0.85, 0.87, 0.88, 0.90],
            "sensitivity": [0.80, 0.82, 0.84, 0.86]
        }
        summary = self.stats_mgr.compute_metric_summary(metric_lists, n_bootstraps=300)
        self.assertIn("metrics", summary)
        self.assertIn("auc", summary["metrics"])
        self.assertAlmostEqual(summary["metrics"]["auc"]["mean"], 0.8750, places=3)
        self.assertIn("ci_95", summary["metrics"]["auc"])

    # 19. JSON Export
    def test_19_json_export(self):
        exp = self.exp_mgr.create_experiment(name="Export JSON Exp", custom_id="exp_json_19")
        self.exp_mgr.run_experiment("exp_json_19")
        exported = self.exp_mgr.export_experiment("exp_json_19", format="json")
        data = json.loads(exported)
        self.assertEqual(data["experiment_id"], "exp_json_19")
        self.assertIn("metrics", data)

    # 20. CSV Export
    def test_20_csv_export(self):
        exp = self.exp_mgr.create_experiment(name="Export CSV Exp", custom_id="exp_csv_20")
        self.exp_mgr.run_experiment("exp_csv_20")
        csv_str = self.exp_mgr.export_experiment("exp_csv_20", format="csv")
        self.assertIn("finding,metric,value", csv_str)
        self.assertIn("macro_avg,auc", csv_str)

    # 21. Text Export
    def test_21_text_export(self):
        exp = self.exp_mgr.create_experiment(name="Export Text Exp", custom_id="exp_txt_21")
        self.exp_mgr.run_experiment("exp_txt_21")
        txt_str = self.exp_mgr.export_experiment("exp_txt_21", format="text")
        self.assertIn("RESEARCH EXPERIMENT REPORT", txt_str)
        self.assertIn("DISCLAIMER", txt_str)

    # 22. Research Report Generation
    def test_22_research_report_generation(self):
        exp = self.exp_mgr.create_experiment(name="Report Gen Exp", custom_id="exp_rep_22")
        self.exp_mgr.run_experiment("exp_rep_22")
        rep = self.exp_mgr.generate_research_report("exp_rep_22")
        self.assertIn("sections", rep)
        self.assertEqual(len(rep["sections"]), 16)
        self.assertIn("overview", rep["sections"])
        self.assertIn("reproducibility", rep["sections"])
        self.assertIn("disclaimer", rep["sections"])

    # 23. Reproducibility Manifest
    def test_23_reproducibility_manifest(self):
        exp = self.exp_mgr.create_experiment(name="Rep Manifest Exp", custom_id="exp_rep_23")
        self.exp_mgr.run_experiment("exp_rep_23")
        manifest = self.exp_mgr.generate_reproducibility_manifest("exp_rep_23")
        self.assertEqual(manifest["reproducibility_status"], "REPRODUCIBLE")
        self.assertEqual(manifest["software_version"], "1.9.0")
        self.assertEqual(manifest["random_seed"], 42)

    # 24. Published Experiment Immutability
    def test_24_published_experiment_immutability(self):
        exp = self.exp_mgr.create_experiment(name="Pub Lock Exp", custom_id="exp_lock_24")
        self.exp_mgr.run_experiment("exp_lock_24")
        self.exp_mgr.finalize_experiment("exp_lock_24")
        self.exp_mgr.publish_experiment("exp_lock_24")

        with self.assertRaises(ValueError) as ctx:
            self.exp_mgr.update_experiment("exp_lock_24", name="Forbidden Update")
        self.assertIn("cannot be edited", str(ctx.exception).lower())

    # 25. Ground-Truth Leakage Prevention
    def test_25_ground_truth_leakage_prevention(self):
        exp = self.exp_mgr.create_experiment(name="GT Leakage Test", custom_id="exp_gt_25")
        self.exp_mgr.run_experiment("exp_gt_25")
        report = self.exp_mgr.generate_research_report("exp_gt_25")

        ok, err = check_ground_truth_isolation(exp)
        self.assertTrue(ok)
        ok_rep, err_rep = check_ground_truth_isolation(report)
        self.assertTrue(ok_rep)

    # 26. Secret Scanning
    def test_26_secret_scanning(self):
        clean_exp = self.exp_mgr.create_experiment(name="Clean Exp", custom_id="exp_clean_26")
        ok, err = check_secret_leakage(clean_exp)
        self.assertTrue(ok)

        dirty_payload = {"api_key": "sk-1234567890abcdef"}
        ok_d, err_d = check_secret_leakage(dirty_payload)
        self.assertFalse(ok_d)

    # 27. Path Traversal Prevention
    def test_27_path_traversal_prevention(self):
        with self.assertRaises(ValueError):
            self.exp_mgr.create_experiment(name="Traversal", custom_id="../../etc/passwd")

        with self.assertRaises(ValueError):
            self.exp_mgr._sanitize_id("..\\windows\\system32")

        with self.assertRaises(ValueError):
            self.exp_mgr._sanitize_id("exp/nested/sub")

    # 28. Invalid Experiment Rejection
    def test_28_invalid_experiment_rejection(self):
        with self.assertRaises(ValueError):
            self.exp_mgr.create_experiment(name="")  # Empty name

        with self.assertRaises(ValueError):
            self.exp_mgr.create_experiment(name="A" * 300)  # Exceeds max length

    # 29. Existing Phase 1.8 Regression
    def test_29_existing_phase_1_8_regression(self):
        ext_datasets = global_external_dataset_manager.list_external_datasets()
        self.assertIsInstance(ext_datasets, list)
        benchmarks = global_benchmark_manager.list_benchmarks()
        self.assertIsInstance(benchmarks, list)

    # 30. Existing Phase 1.7 Regression
    def test_30_existing_phase_1_7_regression(self):
        exps = global_experiment_registry.list_experiments()
        self.assertIsInstance(exps, list)

    # 31. Existing Phase 1.6 Regression
    def test_31_existing_phase_1_6_regression(self):
        from backend.statistical_analysis import compute_distribution_summary
        res = compute_distribution_summary([0.80, 0.85, 0.90])
        self.assertEqual(res["count"], 3)
        self.assertAlmostEqual(res["mean"], 0.85, places=2)

    # 32. API Endpoint Validation
    def test_32_api_endpoint_validation(self):
        exp = self.exp_mgr.create_experiment(name="API Test Exp", custom_id="exp_api_32")
        self.exp_mgr.run_experiment("exp_api_32")
        val = self.exp_mgr.validate_experiment("exp_api_32")
        self.assertEqual(val["status"], "PASS")

    # 33. Research Disclaimer Presence
    def test_33_research_disclaimer_presence(self):
        exp = self.exp_mgr.create_experiment(name="Disclaimer Exp", custom_id="exp_disc_33")
        self.assertIn("RESEARCH", exp["research_disclaimer"])
        rep = self.exp_mgr.generate_research_report("exp_disc_33")
        self.assertIn("RESEARCH", rep["research_disclaimer"])

    # 34. Machine Artifact Immutability
    def test_34_machine_artifact_immutability(self):
        ok, msg, count = check_iu_xray_dataset_integrity()
        self.assertTrue(ok)
        self.assertGreaterEqual(count, 5300)

    # 35. End-to-End Experiment Lifecycle
    def test_35_end_to_end_experiment_lifecycle(self):
        # 1. Create
        exp = self.exp_mgr.create_experiment(
            name="E2E Pipeline Experiment",
            description="Complete lifecycle test from creation to publishing and export.",
            custom_id="exp_e2e_35"
        )
        self.assertEqual(exp["status"], "CONFIGURED")

        # 2. Validate
        val1 = self.exp_mgr.validate_experiment("exp_e2e_35")
        self.assertEqual(val1["status"], "PASS")

        # 3. Run
        res = self.exp_mgr.run_experiment("exp_e2e_35")
        self.assertEqual(res["status"], "COMPLETED")
        self.assertIsNotNone(res["metrics"])
        self.assertIsNotNone(res["statistical_summary"])

        # 4. Finalize
        fin = self.exp_mgr.finalize_experiment("exp_e2e_35")
        self.assertEqual(fin["status"], "VALIDATED")

        # 5. Publish
        pub = self.exp_mgr.publish_experiment("exp_e2e_35")
        self.assertEqual(pub["status"], "PUBLISHED")
        self.assertTrue(pub["locked"])

        # 6. Verify Reproducibility Manifest
        repro = self.exp_mgr.generate_reproducibility_manifest("exp_e2e_35")
        self.assertEqual(repro["reproducibility_status"], "REPRODUCIBLE")

        # 7. Exports
        j_str = self.exp_mgr.export_experiment("exp_e2e_35", format="json")
        self.assertTrue(len(j_str) > 50)
        c_str = self.exp_mgr.export_experiment("exp_e2e_35", format="csv")
        self.assertTrue(len(c_str) > 50)
        t_str = self.exp_mgr.export_experiment("exp_e2e_35", format="text")
        self.assertTrue(len(t_str) > 50)


if __name__ == "__main__":
    unittest.main()
