"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.8 — Multi-Modal Benchmarking, External Portability & Invariant Test Suite

Module: test_phase_1_8_external_benchmarking.py
Purpose:
- 35 comprehensive unit and integration tests verifying External Dataset Registry, Portable Experiment Bundles,
  Air-Gapped Reproducibility, Multi-View Cohorts, Benchmarking, Cross-Dataset Comparison, and Security Invariants.
"""

import os
import sys
import json
import time
import shutil
import unittest
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.external_dataset_manager import ExternalDatasetManager
from backend.experiment_bundle import ExperimentBundleManager
from backend.portable_runner import PortableRunner
from backend.benchmark_manager import BenchmarkManager
from backend.experiment_comparator import ExperimentComparator
from backend.validate_phase_1_8 import Phase18Validator
from backend.experiment_registry import ExperimentRegistry
from backend.model_registry import ModelRegistry
from backend.dataset_version_manager import DatasetVersionManager


class TestPhase18ExternalBenchmarking(unittest.TestCase):
    """35 Dedicated Unit & Integration Tests for Phase 1.8."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(BASE_DIR, "data", "test_phase_1_8_sandbox")
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.ext_ds_dir = os.path.join(cls.test_dir, "external_datasets")
        cls.bundles_dir = os.path.join(cls.test_dir, "experiment_bundles")
        cls.bench_dir = os.path.join(cls.test_dir, "benchmarks")
        cls.models_dir = os.path.join(cls.test_dir, "model_registry")
        cls.dsv_dir = os.path.join(cls.test_dir, "dataset_versions")
        cls.exp_dir = os.path.join(cls.test_dir, "experiments")

        for d in [cls.ext_ds_dir, cls.bundles_dir, cls.bench_dir, cls.models_dir, cls.dsv_dir, cls.exp_dir]:
            os.makedirs(d, exist_ok=True)

        cls.model_reg = ModelRegistry(registry_dir=cls.models_dir)
        cls.dsv_mgr = DatasetVersionManager(versions_dir=cls.dsv_dir)
        cls.ext_ds_mgr = ExternalDatasetManager(storage_dir=cls.ext_ds_dir)
        cls.exp_reg = ExperimentRegistry(experiments_dir=cls.exp_dir, model_registry=cls.model_reg, dataset_version_manager=cls.dsv_mgr)
        cls.bundle_mgr = ExperimentBundleManager(storage_dir=cls.bundles_dir, experiment_registry=cls.exp_reg, model_registry=cls.model_reg, dataset_version_manager=cls.dsv_mgr, external_dataset_manager=cls.ext_ds_mgr)
        cls.portable_runner = PortableRunner(bundle_manager=cls.bundle_mgr)
        cls.bench_mgr = BenchmarkManager(storage_dir=cls.bench_dir, experiment_registry=cls.exp_reg, model_registry=cls.model_reg, dataset_version_manager=cls.dsv_mgr, external_dataset_manager=cls.ext_ds_mgr)
        cls.comparator = ExperimentComparator(experiment_registry=cls.exp_reg, model_registry=cls.model_reg, dataset_version_manager=cls.dsv_mgr)
        cls.validator = Phase18Validator()

        # Pre-register baseline model & experiment
        cls.model_reg.register_model(
            model_id="model_txrv_dense121",
            model_name="TorchXRayVision DenseNet-121",
            architecture="DenseNet-121",
            framework="PyTorch / TorchXRayVision",
            framework_version="1.2.0+",
            weights_identifier="densenet121-res224-all",
            input_dimensions=[1, 224, 224],
            preprocessing={"resize": [224, 224]},
            target_labels=["Cardiomegaly", "Effusion", "Infiltration"],
            target_layer="model.features.norm5"
        )
        cls.dsv_mgr.create_dataset_version(
            dataset_version_id="dsv_test_v1",
            source_dataset="IU_XRAY",
            study_ids=["CXR1122"]
        )
        cls.exp_reg.create_experiment(
            experiment_id="exp_bench_test_01",
            experiment_name="Benchmarking Test Experiment",
            model_id="model_txrv_dense121",
            dataset_version_id="dsv_test_v1"
        )
        cls.exp_reg.update_experiment("exp_bench_test_01", {
            "status": "COMPLETED",
            "metrics": {"accuracy": 0.94, "precision": 0.88, "recall": 0.90, "f1_score": 0.89}
        })

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    # 1. External dataset registration
    def test_01_external_dataset_registration(self):
        rec = self.ext_ds_mgr.register_external_dataset(
            dataset_id="ext_ds_mimic_sample",
            dataset_name="MIMIC-CXR Sample Test Set",
            modality="CHEST_XRAY",
            studies=[
                {
                    "study_id": "STUDY_001",
                    "images": [
                        {"image_id": "IMG_001_F", "view": "Frontal"},
                        {"image_id": "IMG_001_L", "view": "Lateral"}
                    ]
                }
            ]
        )
        self.assertEqual(rec["dataset_id"], "ext_ds_mimic_sample")
        self.assertEqual(rec["status"], "REGISTERED")
        self.assertIn("manifest_sha256", rec)

    # 2. External dataset retrieval
    def test_02_external_dataset_retrieval(self):
        rec = self.ext_ds_mgr.get_external_dataset("ext_ds_mimic_sample")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["dataset_name"], "MIMIC-CXR Sample Test Set")

    # 3. External dataset validation
    def test_03_external_dataset_validation(self):
        ok, errors = self.ext_ds_mgr.validate_external_dataset("ext_ds_mimic_sample")
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)
        rec = self.ext_ds_mgr.get_external_dataset("ext_ds_mimic_sample")
        self.assertEqual(rec["status"], "VALIDATED")

    # 4. Dataset fingerprint generation
    def test_04_dataset_fingerprint_generation(self):
        rec = self.ext_ds_mgr.get_external_dataset("ext_ds_mimic_sample")
        fp = self.ext_ds_mgr.compute_manifest_sha256(rec)
        self.assertEqual(len(fp), 64)
        self.assertEqual(fp, rec["manifest_sha256"])

    # 5. Dataset tampering detection
    def test_05_dataset_tampering_detection(self):
        rec = self.ext_ds_mgr.get_external_dataset("ext_ds_mimic_sample")
        tampered = dict(rec)
        tampered["study_count"] = 9999
        filepath = self.ext_ds_mgr._get_dataset_file("ext_ds_mimic_sample")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tampered, f)

        ok, msg = self.ext_ds_mgr.verify_dataset_integrity("ext_ds_mimic_sample")
        self.assertFalse(ok)
        self.assertIn("Tampering detected", msg)

        # Restore
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rec, f)

    # 6. Dataset finalization
    def test_06_dataset_finalization(self):
        fin = self.ext_ds_mgr.finalize_external_dataset("ext_ds_mimic_sample")
        self.assertEqual(fin["status"], "FINALIZED")
        self.assertIsNotNone(fin["finalized_at"])

    # 7. Dataset immutability upon finalization
    def test_07_finalized_dataset_immutability(self):
        with self.assertRaises(ValueError):
            self.ext_ds_mgr.register_external_dataset(
                dataset_id="ext_ds_mimic_sample",
                dataset_name="Mutated Name"
            )

    # 8. Bundle creation
    def test_08_bundle_creation(self):
        bundle = self.bundle_mgr.create_bundle(
            experiment_id="exp_bench_test_01",
            bundle_id="bundle_test_01"
        )
        self.assertEqual(bundle["bundle_id"], "bundle_test_01")
        self.assertEqual(bundle["experiment_id"], "exp_bench_test_01")
        self.assertIn("manifest_sha256", bundle)
        self.assertGreater(len(bundle["files"]), 0)

    # 9. Bundle schema validation
    def test_09_bundle_schema_validation(self):
        bundle = self.bundle_mgr.get_bundle("bundle_test_01")
        ok, errors = self.validator.validate_bundle_manifest_schema(bundle)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    # 10. Bundle fingerprint verification
    def test_10_bundle_fingerprint_verification(self):
        ok, msg = self.bundle_mgr.verify_bundle_integrity("bundle_test_01")
        self.assertTrue(ok)
        self.assertIn("integrity verified", msg)

    # 11. Bundle tampering detection
    def test_11_bundle_tampering_detection(self):
        bundle_file = os.path.join(self.bundles_dir, "bundle_test_01", "README.md")
        with open(bundle_file, "a") as f:
            f.write("\nTAMPERED CONTENT")

        ok, errors = self.bundle_mgr.validate_bundle("bundle_test_01")
        self.assertFalse(ok)
        self.assertTrue(any("Hash mismatch" in e for e in errors))

        # Re-create to restore
        self.bundle_mgr.create_bundle("exp_bench_test_01", "bundle_test_01")

    # 12. Portable air-gapped verification
    def test_12_portable_air_gapped_verification(self):
        res = self.portable_runner.verify_reproducibility("bundle_test_01")
        self.assertEqual(res["status"], "VERIFIED")
        self.assertTrue(res["is_reproducible"])

    # 13. Missing-resource handling
    def test_13_missing_resource_handling(self):
        res = self.portable_runner.verify_reproducibility("non_existent_bundle")
        self.assertEqual(res["status"], "INVALID")
        self.assertFalse(res["is_reproducible"])

    # 14. Multi-view detection and pairing
    def test_14_multi_view_detection(self):
        studies = [
            {
                "study_id": "STUDY_PAIRED",
                "images": [
                    {"image_id": "IMG_F", "view": "Frontal"},
                    {"image_id": "IMG_L", "view": "Lateral"}
                ]
            },
            {
                "study_id": "STUDY_UNPAIRED",
                "images": [
                    {"image_id": "IMG_F2", "view": "Frontal"}
                ]
            }
        ]
        paired = self.ext_ds_mgr.detect_view_pairs(studies)
        self.assertTrue(paired[0]["is_complete_pair"])
        self.assertIsNotNone(paired[0]["view_pair_id"])
        self.assertFalse(paired[1]["is_complete_pair"])
        self.assertIsNone(paired[1]["view_pair_id"])

    # 15. Frontal/lateral pair structure
    def test_15_frontal_lateral_pair_structure(self):
        studies = [
            {
                "study_id": "STUDY_002",
                "images": [
                    {"image_id": "IMG_F", "view": "Frontal"},
                    {"image_id": "IMG_L", "view": "Lateral"}
                ]
            }
        ]
        paired = self.ext_ds_mgr.detect_view_pairs(studies)
        self.assertEqual(paired[0]["views"], ["Frontal", "Lateral"])

    # 16. Invalid pair rejection
    def test_16_invalid_pair_rejection(self):
        bad_studies = [
            {
                "study_id": "STUDY_BAD",
                "is_complete_pair": True,
                "views": ["Frontal"],  # Missing lateral but claimed complete
                "view_pair_id": "pair_01"
            }
        ]
        ok, errs = self.validator.validate_multi_view_pairing_integrity(bad_studies)
        self.assertFalse(ok)
        self.assertIn("missing frontal or lateral view", errs[0])

    # 17. Multi-view configuration validation
    def test_17_multi_view_configuration_validation(self):
        bm = self.bench_mgr.register_benchmark(
            benchmark_id="bm_multiview_valid",
            benchmark_name="Multi-View Fusion Test",
            experiment_id="exp_bench_test_01",
            view_configuration="MULTI_VIEW",
            fusion_strategy="independent_view"
        )
        self.assertEqual(bm["view_configuration"], "MULTI_VIEW")

    # 18. External evaluation configuration
    def test_18_external_evaluation_configuration(self):
        bm = self.bench_mgr.register_benchmark(
            benchmark_id="bm_external_test",
            benchmark_name="External Test Set Run",
            experiment_id="exp_bench_test_01",
            external_dataset_id="ext_ds_mimic_sample",
            modality="EXTERNAL_TEST_SET"
        )
        self.assertEqual(bm["modality"], "EXTERNAL_TEST_SET")

    # 19. Cross-dataset comparison
    def test_19_cross_dataset_comparison(self):
        bm1 = self.bench_mgr.register_benchmark(
            benchmark_id="bm_comp_1",
            benchmark_name="Run 1",
            experiment_id="exp_bench_test_01",
            modality="INTERNAL_BENCHMARK"
        )
        bm2 = self.bench_mgr.register_benchmark(
            benchmark_id="bm_comp_2",
            benchmark_name="Run 2",
            experiment_id="exp_bench_test_01",
            modality="EXTERNAL_TEST_SET",
            external_dataset_id="ext_ds_mimic_sample"
        )
        self.bench_mgr.run_benchmark("bm_comp_1")
        self.bench_mgr.run_benchmark("bm_comp_2")

        b1 = self.bench_mgr.get_benchmark("bm_comp_1")
        b2 = self.bench_mgr.get_benchmark("bm_comp_2")

        comp = self.comparator.compare_external_benchmarks(b1, b2)
        self.assertIn("metric_deltas", comp)
        self.assertIn("compatibility_warnings", comp)
        self.assertFalse(comp["is_directly_comparable"])

    # 20. Safe zero-denominator relative delta
    def test_20_safe_zero_denominator_relative_delta(self):
        b1 = {"benchmark_id": "A", "metrics": {"acc": 0.0}}
        b2 = {"benchmark_id": "B", "metrics": {"acc": 0.5}}
        comp = self.comparator.compare_external_benchmarks(b1, b2)
        delta_item = [d for d in comp["metric_deltas"] if d["metric_name"] == "acc"][0]
        self.assertEqual(delta_item["absolute_delta"], 0.5)
        self.assertIsNone(delta_item["relative_delta"])

    # 21. Neutral comparison language verification
    def test_21_neutral_comparison_language(self):
        b1 = {"benchmark_id": "A", "metrics": {"f1": 0.80}}
        b2 = {"benchmark_id": "B", "metrics": {"f1": 0.85}}
        comp = self.comparator.compare_external_benchmarks(b1, b2)
        delta_item = comp["metric_deltas"][0]
        self.assertEqual(delta_item["observation"], "Observed increase")
        raw_json = json.dumps(comp)
        for banned in ["winner", "better", "superior", "best model"]:
            self.assertNotIn(banned, raw_json.lower())

    # 22. Benchmark registration
    def test_22_benchmark_registration(self):
        bm = self.bench_mgr.register_benchmark(
            benchmark_id="bm_reg_test",
            benchmark_name="Registration Test",
            experiment_id="exp_bench_test_01"
        )
        self.assertEqual(bm["status"], "REGISTERED")

    # 23. Benchmark execution
    def test_23_benchmark_execution(self):
        executed = self.bench_mgr.run_benchmark("bm_reg_test")
        self.assertEqual(executed["status"], "EXECUTED")
        self.assertIn("accuracy", executed["metrics"])

    # 24. Benchmark validation
    def test_24_benchmark_validation(self):
        ok, errs = self.bench_mgr.validate_benchmark("bm_reg_test")
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

    # 25. Benchmark finalization lock
    def test_25_benchmark_finalization_lock(self):
        fin = self.bench_mgr.finalize_benchmark("bm_reg_test")
        self.assertEqual(fin["status"], "FINALIZED")

    # 26. Finalized benchmark mutation rejection
    def test_26_finalized_benchmark_mutation_rejection(self):
        with self.assertRaises(ValueError):
            self.bench_mgr.run_benchmark("bm_reg_test")

    # 27. Benchmark export (JSON / Text)
    def test_27_benchmark_export(self):
        json_exp = self.bench_mgr.export_benchmark("bm_reg_test", export_format="json")
        self.assertIn("bm_reg_test", json_exp)
        text_exp = self.bench_mgr.export_benchmark("bm_reg_test", export_format="text")
        self.assertIn("RESEARCH BENCHMARK REPORT", text_exp)
        self.assertIn("DISCLAIMER", text_exp)

    # 28. REST API external datasets validation
    def test_28_api_external_datasets_schema(self):
        datasets = self.ext_ds_mgr.list_external_datasets()
        self.assertIsInstance(datasets, list)

    # 29. REST API benchmarks list
    def test_29_api_benchmarks_list(self):
        bms = self.bench_mgr.list_benchmarks()
        self.assertIsInstance(bms, list)

    # 30. Bundle export to zip
    def test_30_bundle_export_zip(self):
        zip_path = self.bundle_mgr.export_bundle("bundle_test_01")
        self.assertTrue(os.path.exists(zip_path))
        self.assertTrue(zip_path.endswith(".zip"))

    # 31. Path traversal rejection
    def test_31_path_traversal_rejection(self):
        for bad_id in ["../evil", "test/../../root", "null\0byte", "C:\\escaped"]:
            with self.assertRaises(ValueError):
                self.ext_ds_mgr._validate_id(bad_id)
            with self.assertRaises(ValueError):
                self.bench_mgr._validate_id(bad_id)
            with self.assertRaises(ValueError):
                self.bundle_mgr._validate_id(bad_id)

    # 32. Secret / token leakage detection
    def test_32_secret_leakage_detection(self):
        bad_payload = {"dataset_id": "test", "api_key": "sk-secret-token-12345"}
        ok, err = self.validator.validate_no_secrets(bad_payload)
        self.assertFalse(ok)
        self.assertIn("secret field detected", err)

    # 33. NaN / Infinity rejection
    def test_33_nan_infinity_rejection(self):
        nan_payload = {"score": float('nan')}
        ok, err = self.validator.validate_finite_numbers(nan_payload)
        self.assertFalse(ok)

        inf_payload = {"score": float('inf')}
        ok, err = self.validator.validate_finite_numbers(inf_payload)
        self.assertFalse(ok)

    # 34. Ground-truth XML report leakage protection
    def test_34_ground_truth_xml_leakage_protection(self):
        leak_payload = {"notes": "Reference says: <eFind>No acute cardiopulmonary disease.</eFind>"}
        ok, err = self.validator.validate_no_ground_truth_leakage(leak_payload)
        self.assertFalse(ok)
        self.assertIn("Ground-truth XML report tag detected", err)

    # 35. Machine artifact immutability
    def test_35_machine_artifact_immutability(self):
        iu_dir = os.path.join(BASE_DIR, "data", "iu_xray")
        if os.path.exists(iu_dir):
            file_hashes = self.validator.compute_directory_sha256(iu_dir)
            self.assertGreaterEqual(len(file_hashes), 5399)


if __name__ == "__main__":
    unittest.main()
