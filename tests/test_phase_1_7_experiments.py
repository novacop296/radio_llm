"""
Unit and Integration Test Suite for Phase 1.7 — Reproducible Experiment Registry, Model Versioning & Longitudinal Research Tracking.

Test Coverage:
1. Model registration
2. Model retrieval
3. Model hash validation
4. Model finalization
5. Dataset version creation
6. Dataset version fingerprint
7. Dataset finalization
8. Experiment creation
9. Experiment lifecycle
10. Experiment configuration validation
11. Experiment fingerprint determinism
12. Fingerprint changes when configuration changes
13. Experiment execution
14. Evaluation run attachment
15. Experiment validation
16. Experiment finalization
17. Experiment archive
18. Experiment snapshot creation
19. Snapshot hash validation
20. Snapshot immutability
21. Experiment history
22. Experiment provenance
23. Experiment comparison
24. Zero-denominator comparison handling
25. Incompatible dataset comparison
26. Incompatible methodology comparison
27. NaN rejection
28. Infinity rejection
29. Path traversal rejection
30. Secret leakage detection
31. Ground-truth XML leakage detection
32. Machine artifact immutability
33. Finalized experiment mutation rejection
34. API regression checks
35. Export validation
"""

import os
import sys
import json
import math
import shutil
import tempfile
import unittest
import hashlib
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_registry import ModelRegistry
from dataset_version_manager import DatasetVersionManager
from experiment_registry import ExperimentRegistry
from experiment_runner import ExperimentRunner
from experiment_history import ExperimentHistoryTracker
from experiment_snapshot import ExperimentSnapshotManager
from experiment_comparator import ExperimentComparator
from validate_phase_1_7 import Phase17Validator


class TestPhase17ExperimentRegistry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "model_registry"
        self.dsv_dir = Path(self.temp_dir) / "dataset_versions"
        self.exp_dir = Path(self.temp_dir) / "experiments"
        self.history_dir = Path(self.temp_dir) / "experiments" / "history"
        self.snapshots_dir = Path(self.temp_dir) / "experiment_snapshots"
        self.data_dir = Path(self.temp_dir) / "iu_xray"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dsv_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy study directories in data_dir
        for sid in ["CXR1122", "CXR2001", "CXR3002"]:
            sdir = self.data_dir / sid
            sdir.mkdir(parents=True, exist_ok=True)
            with open(sdir / "evidence_findings.json", "w", encoding="utf-8") as f:
                json.dump({"study_id": sid, "findings": [{"finding": "Cardiomegaly", "status": "possible"}]}, f)

        # Initialize managers with isolated directories
        self.model_reg = ModelRegistry(registry_dir=str(self.models_dir))
        self.dsv_mgr = DatasetVersionManager(versions_dir=str(self.dsv_dir), data_dir=str(self.data_dir))
        self.history_trk = ExperimentHistoryTracker(history_dir=str(self.history_dir))
        self.snap_mgr = ExperimentSnapshotManager(snapshots_dir=str(self.snapshots_dir))
        self.exp_reg = ExperimentRegistry(
            experiments_dir=str(self.exp_dir),
            model_registry=self.model_reg,
            dataset_version_manager=self.dsv_mgr,
            history_tracker=self.history_trk,
            snapshot_manager=self.snap_mgr
        )
        self.exp_runner = ExperimentRunner(
            experiment_registry=self.exp_reg,
            model_registry=self.model_reg,
            dataset_version_manager=self.dsv_mgr,
            snapshot_manager=self.snap_mgr,
            experiments_dir=str(self.exp_dir)
        )
        self.comparator = ExperimentComparator(
            experiment_registry=self.exp_reg,
            model_registry=self.model_reg,
            dataset_version_manager=self.dsv_mgr
        )
        self.validator = Phase17Validator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Model registration
    def test_01_model_registration(self):
        model = self.model_reg.register_model(
            model_id="model_custom_dense",
            model_name="Custom DenseNet Model",
            architecture="DenseNet-121",
            framework="PyTorch",
            framework_version="2.0.0",
            weights_identifier="custom-weights-v1",
            input_dimensions=[1, 224, 224],
            preprocessing={"resize": [224, 224]},
            target_labels=["Cardiomegaly", "Effusion"],
            target_layer="model.features.norm5",
            description="Test custom model",
            registered_by="researcher_alice"
        )
        self.assertEqual(model["model_id"], "model_custom_dense")
        self.assertEqual(model["status"], "REGISTERED")
        self.assertEqual(model["registered_by"], "researcher_alice")
        self.assertIn("RESEARCH MODEL REGISTRY RECORD", model["disclaimer"])

    # 2. Model retrieval
    def test_02_model_retrieval(self):
        self.model_reg.register_model(
            model_id="model_retrieval_test",
            model_name="Retrieval Test",
            architecture="DenseNet-121",
            framework="PyTorch",
            framework_version="2.0.0",
            weights_identifier="ret-weights",
            input_dimensions=[1, 224, 224],
            preprocessing={},
            target_labels=["Pneumonia"],
            target_layer="norm5"
        )
        ret = self.model_reg.get_model("model_retrieval_test")
        self.assertIsNotNone(ret)
        self.assertEqual(ret["model_name"], "Retrieval Test")
        self.assertIsNone(self.model_reg.get_model("model_nonexistent"))

    # 3. Model hash validation
    def test_03_model_hash_validation(self):
        model = self.model_reg.register_model(
            model_id="model_hash_test",
            model_name="Hash Test",
            architecture="DenseNet-121",
            framework="PyTorch",
            framework_version="2.0.0",
            weights_identifier="hash-weights",
            input_dimensions=[1, 224, 224],
            preprocessing={},
            target_labels=["Cardiomegaly"],
            target_layer="norm5",
            weights_sha256="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        ver = self.model_reg.verify_model_hash("model_hash_test")
        self.assertEqual(ver["recorded_sha256"], "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
        self.assertIn("RESEARCH MODEL REGISTRY RECORD", ver["disclaimer"])

    # 4. Model finalization
    def test_04_model_finalization(self):
        self.model_reg.register_model(
            model_id="model_finalize_test",
            model_name="Finalize Test",
            architecture="DenseNet-121",
            framework="PyTorch",
            framework_version="2.0.0",
            weights_identifier="fin-weights",
            input_dimensions=[1, 224, 224],
            preprocessing={},
            target_labels=["Cardiomegaly"],
            target_layer="norm5"
        )
        fin = self.model_reg.finalize_model("model_finalize_test")
        self.assertEqual(fin["status"], "FINALIZED")
        # Attempt to update finalized model should fail
        with self.assertRaises(ValueError):
            self.model_reg.update_model_before_finalization("model_finalize_test", {"model_name": "New Name"})

    # 5. Dataset version creation
    def test_05_dataset_version_creation(self):
        dsv = self.dsv_mgr.create_dataset_version(
            dataset_version_id="dsv_test_cohort_v1",
            source_dataset="IU_XRAY",
            study_ids=["CXR1122", "CXR2001"],
            allowed_reference_annotations=["Cardiomegaly", "Effusion"],
            created_by="curator_bob"
        )
        self.assertEqual(dsv["dataset_version_id"], "dsv_test_cohort_v1")
        self.assertEqual(dsv["study_count"], 2)
        self.assertEqual(dsv["status"], "DRAFT")
        self.assertTrue(len(dsv["manifest_sha256"]) == 64)

    # 6. Dataset version fingerprint
    def test_06_dataset_version_fingerprint(self):
        self.dsv_mgr.create_dataset_version(
            dataset_version_id="dsv_fp_test",
            source_dataset="IU_XRAY",
            study_ids=["CXR1122"],
            allowed_reference_annotations=["Effusion"]
        )
        fp = self.dsv_mgr.fingerprint_dataset_version("dsv_fp_test")
        self.assertEqual(fp["dataset_version_id"], "dsv_fp_test")
        self.assertEqual(fp["study_count"], 1)
        self.assertTrue(len(fp["manifest_sha256"]) == 64)

    # 7. Dataset finalization
    def test_07_dataset_finalization(self):
        self.dsv_mgr.create_dataset_version(
            dataset_version_id="dsv_fin_test",
            source_dataset="IU_XRAY",
            study_ids=["CXR1122", "CXR3002"]
        )
        fin_dsv = self.dsv_mgr.finalize_dataset_version("dsv_fin_test")
        self.assertEqual(fin_dsv["status"], "FINALIZED")
        self.assertIsNotNone(fin_dsv.get("finalized_at"))

    # 8. Experiment creation
    def test_08_experiment_creation(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_creation_test",
            experiment_name="Creation Test Experiment",
            description="Testing registry experiment creation",
            model_id="model_densenet121_txrv",
            dataset_version_id="dsv_test_cohort_v1",
            evaluation_dataset_id="eval_ds_default",
            methodology="standard_qa_evidence_evaluation"
        )
        self.assertEqual(exp["experiment_id"], "exp_creation_test")
        self.assertEqual(exp["status"], "REGISTERED")
        self.assertTrue(len(exp["fingerprint_sha256"]) == 64)
        self.assertEqual(len(exp["provenance_trail"]), 1)

    # 9. Experiment lifecycle
    def test_09_experiment_lifecycle(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_lifecycle_test",
            experiment_name="Lifecycle Experiment"
        )
        self.assertEqual(exp["status"], "REGISTERED")
        # Update config -> CONFIGURED
        updated = self.exp_reg.update_experiment("exp_lifecycle_test", {"random_seed": 99})
        self.assertEqual(updated["status"], "CONFIGURED")
        # Start -> RUNNING
        started = self.exp_reg.start_experiment("exp_lifecycle_test")
        self.assertEqual(started["status"], "RUNNING")
        # Complete -> COMPLETED
        completed = self.exp_reg.complete_experiment(
            "exp_lifecycle_test",
            evaluation_run_id="eval_001",
            metrics={"accuracy": 0.85, "precision": 0.80}
        )
        self.assertEqual(completed["status"], "COMPLETED")
        # Validate -> VALIDATED
        validated = self.exp_reg.validate_experiment("exp_lifecycle_test")
        self.assertEqual(validated["status"], "VALIDATED")

    # 10. Experiment configuration validation
    def test_10_experiment_configuration_validation(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_config_val_test",
            experiment_name="Config Val Test"
        )
        ok, errors = self.validator.validate_experiment_record(exp)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    # 11. Experiment fingerprint determinism
    def test_11_experiment_fingerprint_determinism(self):
        cfg1 = {
            "model_id": "model_densenet121_txrv",
            "dataset_version_id": "dsv_v1",
            "evaluation_dataset_id": "eval_ds_1",
            "methodology": "standard_qa_evidence_evaluation",
            "preprocessing": {"resize": [224, 224]},
            "inference_configuration": {"threshold": 0.15, "top_k": 5},
            "grounding_configuration": {"target_layer": "model.features.norm5"},
            "report_configuration": {"llm_provider": "mock"},
            "random_seed": 42
        }
        cfg2 = dict(cfg1)
        fp1 = self.exp_reg.generate_experiment_fingerprint(cfg1)
        fp2 = self.exp_reg.generate_experiment_fingerprint(cfg2)
        self.assertEqual(fp1, fp2)

    # 12. Fingerprint changes when configuration changes
    def test_12_fingerprint_changes_when_config_changes(self):
        cfg1 = {
            "model_id": "model_densenet121_txrv",
            "dataset_version_id": "dsv_v1",
            "evaluation_dataset_id": "eval_ds_1",
            "methodology": "standard_qa_evidence_evaluation",
            "random_seed": 42
        }
        cfg2 = dict(cfg1)
        cfg2["random_seed"] = 99
        fp1 = self.exp_reg.generate_experiment_fingerprint(cfg1)
        fp2 = self.exp_reg.generate_experiment_fingerprint(cfg2)
        self.assertNotEqual(fp1, fp2)

    # 13. Experiment execution
    def test_13_experiment_execution(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_exec_test",
            experiment_name="Exec Test"
        )
        run_res = self.exp_runner.run_experiment("exp_exec_test")
        self.assertEqual(run_res["status"], "COMPLETED")
        self.assertIsNotNone(run_res.get("metrics"))
        self.assertIsNotNone(run_res.get("statistics"))
        self.assertIsNotNone(run_res.get("error_analysis"))

    # 14. Evaluation run attachment
    def test_14_evaluation_run_attachment(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_attach_test",
            experiment_name="Attach Test"
        )
        completed = self.exp_reg.complete_experiment(
            experiment_id="exp_attach_test",
            evaluation_run_id="eval_run_999",
            metrics={"accuracy": 0.92, "f1_score": 0.88}
        )
        self.assertEqual(completed["evaluation_run_id"], "eval_run_999")
        self.assertEqual(completed["metrics"]["accuracy"], 0.92)

    # 15. Experiment validation
    def test_15_experiment_validation(self):
        self.exp_reg.create_experiment(
            experiment_id="exp_validate_test",
            experiment_name="Validate Test"
        )
        self.exp_runner.run_experiment("exp_validate_test")
        res = self.exp_reg.validate_experiment("exp_validate_test")
        self.assertTrue(res["valid"])
        self.assertEqual(res["status"], "VALIDATED")

    # 16. Experiment finalization
    def test_16_experiment_finalization(self):
        self.exp_reg.create_experiment(
            experiment_id="exp_finalize_test",
            experiment_name="Finalize Test"
        )
        self.exp_runner.run_experiment("exp_finalize_test")
        fin = self.exp_reg.finalize_experiment("exp_finalize_test")
        self.assertEqual(fin["status"], "FINALIZED")
        self.assertIsNotNone(fin.get("snapshot_sha256"))

    # 17. Experiment archive
    def test_17_experiment_archive(self):
        self.exp_reg.create_experiment(
            experiment_id="exp_archive_test",
            experiment_name="Archive Test"
        )
        arch = self.exp_reg.archive_experiment("exp_archive_test")
        self.assertEqual(arch["status"], "ARCHIVED")

    # 18. Experiment snapshot creation
    def test_18_experiment_snapshot_creation(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_snap_create_test",
            experiment_name="Snap Test"
        )
        snap = self.snap_mgr.create_snapshot(
            experiment_id="exp_snap_create_test",
            experiment_config=exp,
            metrics={"accuracy": 0.89}
        )
        self.assertEqual(snap["experiment_id"], "exp_snap_create_test")
        self.assertTrue(len(snap["snapshot_sha256"]) == 64)
        self.assertEqual(snap["manifest"]["metrics"]["accuracy"], 0.89)

    # 19. Snapshot hash validation
    def test_19_snapshot_hash_validation(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_snap_hash_test",
            experiment_name="Snap Hash Test"
        )
        self.snap_mgr.create_snapshot(
            experiment_id="exp_snap_hash_test",
            experiment_config=exp,
            metrics={"accuracy": 0.85}
        )
        ver = self.snap_mgr.verify_snapshot("exp_snap_hash_test")
        self.assertTrue(ver["verified"])
        self.assertEqual(ver["recorded_sha256"], ver["computed_sha256"])

    # 20. Snapshot immutability
    def test_20_snapshot_immutability(self):
        exp = self.exp_reg.create_experiment(
            experiment_id="exp_snap_immut_test",
            experiment_name="Immut Test"
        )
        snap1 = self.snap_mgr.create_snapshot(
            experiment_id="exp_snap_immut_test",
            experiment_config=exp,
            metrics={"accuracy": 0.85}
        )
        # Re-attempting snapshot creation returns exact existing snapshot
        snap2 = self.snap_mgr.create_snapshot(
            experiment_id="exp_snap_immut_test",
            experiment_config=exp,
            metrics={"accuracy": 0.99}  # Modified metrics ignored
        )
        self.assertEqual(snap1["snapshot_sha256"], snap2["snapshot_sha256"])
        self.assertEqual(snap2["manifest"]["metrics"]["accuracy"], 0.85)

    # 21. Experiment history
    def test_21_experiment_history(self):
        self.exp_reg.create_experiment(
            experiment_id="exp_hist_test",
            experiment_name="Hist Test"
        )
        self.history_trk.record_event(
            experiment_id="exp_hist_test",
            action="TEST_ACTION",
            actor="researcher_alice",
            object_type="EXPERIMENT",
            field="parameter_x",
            old_value=1,
            new_value=2
        )
        timeline = self.history_trk.get_timeline("exp_hist_test")
        self.assertGreaterEqual(timeline["total_events"], 2)
        actions = [ev["action"] for ev in timeline["timeline"]]
        self.assertIn("EXPERIMENT_REGISTERED", actions)
        self.assertIn("TEST_ACTION", actions)

    # 22. Experiment provenance
    def test_22_experiment_provenance(self):
        self.exp_reg.create_experiment(
            experiment_id="exp_prov_test",
            experiment_name="Prov Test"
        )
        prov = self.exp_reg.get_experiment_provenance("exp_prov_test")
        self.assertIsInstance(prov, list)
        self.assertGreaterEqual(len(prov), 1)
        self.assertEqual(prov[0]["stage"], "EXPERIMENT_REGISTRATION")

    # 23. Experiment comparison
    def test_23_experiment_comparison(self):
        exp1 = self.exp_reg.create_experiment(
            experiment_id="exp_cmp_1",
            experiment_name="Exp A",
            inference_configuration={"threshold": 0.15}
        )
        self.exp_reg.complete_experiment("exp_cmp_1", metrics={"accuracy": 0.80, "recall": 0.70})

        exp2 = self.exp_reg.create_experiment(
            experiment_id="exp_cmp_2",
            experiment_name="Exp B",
            inference_configuration={"threshold": 0.25}
        )
        self.exp_reg.complete_experiment("exp_cmp_2", metrics={"accuracy": 0.85, "recall": 0.65})

        comp = self.comparator.compare_experiments(["exp_cmp_1", "exp_cmp_2"])
        self.assertIn("metric_deltas", comp)
        self.assertAlmostEqual(comp["metric_deltas"]["accuracy"], 0.05, places=4)
        self.assertAlmostEqual(comp["metric_deltas"]["recall"], -0.05, places=4)
        self.assertIn("RESEARCH EVALUATION ONLY", comp["evaluation_notice"])

    # 24. Zero-denominator comparison handling
    def test_24_zero_denominator_comparison_handling(self):
        exp1 = self.exp_reg.create_experiment("exp_zd_1", "Exp ZD1")
        self.exp_reg.complete_experiment("exp_zd_1", metrics={"custom_rate": 0.0})

        exp2 = self.exp_reg.create_experiment("exp_zd_2", "Exp ZD2")
        self.exp_reg.complete_experiment("exp_zd_2", metrics={"custom_rate": 0.10})

        comp = self.comparator.compare_experiments(["exp_zd_1", "exp_zd_2"])
        self.assertEqual(comp["metric_deltas"]["custom_rate"], 0.10)
        self.assertIsNone(comp["relative_deltas"]["custom_rate"])

    # 25. Incompatible dataset comparison
    def test_25_incompatible_dataset_comparison(self):
        exp1 = self.exp_reg.create_experiment("exp_ds_a", "Exp A", dataset_version_id="dsv_cohort_a")
        self.exp_reg.complete_experiment("exp_ds_a", metrics={"accuracy": 0.85})

        exp2 = self.exp_reg.create_experiment("exp_ds_b", "Exp B", dataset_version_id="dsv_cohort_b")
        self.exp_reg.complete_experiment("exp_ds_b", metrics={"accuracy": 0.90})

        comp = self.comparator.compare_experiments(["exp_ds_a", "exp_ds_b"])
        self.assertFalse(comp["is_directly_comparable"])
        self.assertFalse(comp["dataset_compatibility"]["compatible"])

    # 26. Incompatible methodology comparison
    def test_26_incompatible_methodology_comparison(self):
        exp1 = self.exp_reg.create_experiment("exp_meth_a", "Exp A", methodology="method_v1")
        self.exp_reg.complete_experiment("exp_meth_a", metrics={"accuracy": 0.85})

        exp2 = self.exp_reg.create_experiment("exp_meth_b", "Exp B", methodology="method_v2")
        self.exp_reg.complete_experiment("exp_meth_b", metrics={"accuracy": 0.85})

        comp = self.comparator.compare_experiments(["exp_meth_a", "exp_meth_b"])
        self.assertFalse(comp["is_directly_comparable"])
        self.assertFalse(comp["methodology_compatibility"]["compatible"])

    # 27. NaN rejection
    def test_27_nan_rejection(self):
        payload_with_nan = {"metric_val": float("nan")}
        ok, err = self.validator.validate_finite_numbers(payload_with_nan)
        self.assertFalse(ok)
        self.assertIn("non-finite float", err)

    # 28. Infinity rejection
    def test_28_infinity_rejection(self):
        payload_with_inf = {"metric_val": float("inf")}
        ok, err = self.validator.validate_finite_numbers(payload_with_inf)
        self.assertFalse(ok)
        self.assertIn("non-finite float", err)

    # 29. Path traversal rejection
    def test_29_path_traversal_rejection(self):
        with self.assertRaises(ValueError):
            self.model_reg.get_model("../secret_model")
        with self.assertRaises(ValueError):
            self.dsv_mgr.get_dataset_version("../../etc/passwd")
        with self.assertRaises(ValueError):
            self.exp_reg.get_experiment("..\\windows\\system32")

    # 30. Secret leakage detection
    def test_30_secret_leakage_detection(self):
        leaked_payload = {"api_key": "sk-123456789012345678901234567890"}
        ok, err = self.validator.validate_no_secrets(leaked_payload)
        self.assertFalse(ok)
        self.assertIn("Secret credential", err)

    # 31. Ground-truth XML leakage detection
    def test_31_ground_truth_xml_leakage_detection(self):
        leaked_payload = {"notes": "Found <eFind>No acute cardiopulmonary disease</eFind>"}
        ok, err = self.validator.validate_no_ground_truth_leakage(leaked_payload)
        self.assertFalse(ok)
        self.assertIn("Ground-truth XML tag", err)

    # 32. Machine artifact immutability
    def test_32_machine_artifact_immutability(self):
        hashes_before = {}
        for root, _, files in os.walk(str(self.data_dir)):
            for fn in files:
                fp = os.path.join(root, fn)
                with open(fp, "rb") as bf:
                    hashes_before[fp] = hashlib.sha256(bf.read()).hexdigest()

        # Run multiple experiments
        self.exp_reg.create_experiment("exp_immut_test_1", "Exp Immut 1")
        self.exp_runner.run_experiment("exp_immut_test_1")
        self.exp_reg.finalize_experiment("exp_immut_test_1")

        hashes_after = {}
        for root, _, files in os.walk(str(self.data_dir)):
            for fn in files:
                fp = os.path.join(root, fn)
                with open(fp, "rb") as bf:
                    hashes_after[fp] = hashlib.sha256(bf.read()).hexdigest()

        self.assertEqual(hashes_before, hashes_after)

    # 33. Finalized experiment mutation rejection
    def test_33_finalized_experiment_mutation_rejection(self):
        exp = self.exp_reg.create_experiment("exp_mut_test", "Mut Test")
        self.exp_runner.run_experiment("exp_mut_test")
        self.exp_reg.finalize_experiment("exp_mut_test")

        # Attempt to update finalized experiment must raise ValueError
        with self.assertRaises(ValueError):
            self.exp_reg.update_experiment("exp_mut_test", {"description": "Modified description"})

    # 34. API regression checks
    def test_34_api_models_and_dataset_versions_list(self):
        models = self.model_reg.list_models()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)  # Default DenseNet bootstrap

        dsvs = self.dsv_mgr.list_dataset_versions()
        self.assertIsInstance(dsvs, list)

        exps = self.exp_reg.list_experiments()
        self.assertIsInstance(exps, list)

    # 35. Export validation
    def test_35_export_validation(self):
        self.exp_reg.create_experiment("exp_exp_pkg_test", "Export Test")
        self.exp_runner.run_experiment("exp_exp_pkg_test")

        json_str, mime_json = self.exp_runner.generate_export_package("exp_exp_pkg_test", format_type="json")
        self.assertEqual(mime_json, "application/json")
        parsed = json.loads(json_str)
        self.assertEqual(parsed["experiment_id"], "exp_exp_pkg_test")

        text_str, mime_text = self.exp_runner.generate_export_package("exp_exp_pkg_test", format_type="text")
        self.assertEqual(mime_text, "text/plain")
        self.assertIn("EXPLAINABLE RADIOLOGY RESEARCH EXPERIMENT REPORT", text_str)
        self.assertIn("MANDATORY RESEARCH DISCLAIMER", text_str)


if __name__ == "__main__":
    unittest.main()
