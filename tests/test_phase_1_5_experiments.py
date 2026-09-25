"""
Unit and Integration Test Suite for Phase 1.5 — Research Evaluation, Experiment Tracking & Dataset Comparison.

Validates:
1. Immutable Dataset Snapshots (manifest hashing, artifact checksums, storage isolation).
2. Configuration Fingerprinting (SHA-256 determinism, secret exclusion, canonical hashing).
3. Experiment Lifecycle (CREATED -> RUNNING -> COMPLETED -> ARCHIVED, immutability on completed).
4. Research Evaluation Metrics (bounded ratios, Kappa metrics, no NaN/Infinity, non-evaluative terminology).
5. 11-Stage Experiment Provenance Pipeline.
6. Multi-Experiment Side-by-Side Comparison (config diffs, metric deltas without rankings/scores).
7. Invariant Validation (ground-truth leakage protection, zero secrets, zero path traversal).
8. Phase 1.5 REST API endpoints and regression compatibility.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dataset_snapshot_manager import DatasetSnapshotManager
from experiment_manager import ExperimentManager
from experiment_runner import ExperimentRunner
from experiment_comparator import ExperimentComparator
from research_metrics import ResearchMetricsCalculator
from validate_experiment import ExperimentValidator


class TestDatasetSnapshotManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.snapshots_dir = Path(self.temp_dir) / "snapshots"
        self.iu_xray_dir = Path(self.temp_dir) / "iu_xray"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.iu_xray_dir.mkdir(parents=True, exist_ok=True)

        for study_id in ["CXR1122", "CXR2001", "CXR3002"]:
            s_dir = self.iu_xray_dir / study_id
            s_dir.mkdir(parents=True, exist_ok=True)
            with open(s_dir / "evidence_findings.json", "w") as f:
                json.dump({"study_id": study_id, "findings": [{"finding": "Cardiomegaly", "status": "possible"}]}, f)

        self.snapshot_mgr = DatasetSnapshotManager(
            snapshots_dir=str(self.snapshots_dir),
            data_dir=str(self.iu_xray_dir)
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_snapshot_manifest_and_immutability(self):
        snap = self.snapshot_mgr.create_snapshot(
            name="Test Snapshot",
            description="Unit test snapshot",
            study_ids=["CXR1122", "CXR2001"],
            created_by="test_curator"
        )
        self.assertIn("snapshot_id", snap)
        self.assertEqual(snap["study_count"], 2)
        self.assertEqual(len(snap["manifest_sha256"]), 64)
        self.assertEqual(len(snap["study_ids"]), 2)

        # Verify file exists on disk
        snap_path = self.snapshots_dir / f"{snap['snapshot_id']}.json"
        self.assertTrue(snap_path.exists())

    def test_snapshot_manifest_hash_reproducibility(self):
        snap1 = self.snapshot_mgr.create_snapshot(
            name="Snap A",
            description="Reproducibility test",
            study_ids=["CXR1122", "CXR2001"],
            custom_id="snap_repro_1"
        )
        snap2 = self.snapshot_mgr.create_snapshot(
            name="Snap B",
            description="Reproducibility test 2",
            study_ids=["CXR2001", "CXR1122"],  # Reversed order
            custom_id="snap_repro_2"
        )
        # Manifest hashes should match regardless of input ordering
        self.assertEqual(snap1["manifest_sha256"], snap2["manifest_sha256"])

    def test_snapshot_empty_study_ids_rejection(self):
        with self.assertRaises(ValueError):
            self.snapshot_mgr.create_snapshot(
                name="Empty Snap",
                description="Should fail",
                study_ids=[]
            )

    def test_list_and_get_snapshots(self):
        s1 = self.snapshot_mgr.create_snapshot(name="Snap 1", description="1", study_ids=["CXR1122"])
        s2 = self.snapshot_mgr.create_snapshot(name="Snap 2", description="2", study_ids=["CXR2001"])

        snapshots = self.snapshot_mgr.list_snapshots()
        self.assertGreaterEqual(len(snapshots), 2)

        fetched = self.snapshot_mgr.get_snapshot(s1["snapshot_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["snapshot_id"], s1["snapshot_id"])
        self.assertEqual(fetched["name"], "Snap 1")

    def test_validate_snapshot_integrity(self):
        s = self.snapshot_mgr.create_snapshot(name="Integrity Snap", description="Check", study_ids=["CXR1122"])
        is_valid, errors = self.snapshot_mgr.validate_snapshot_integrity(s["snapshot_id"])
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_snapshot_storage_isolation(self):
        self.assertNotEqual(str(self.snapshots_dir), str(self.iu_xray_dir))
        snap = self.snapshot_mgr.create_snapshot(name="Isolated", description="Isolation test", study_ids=["CXR1122"])
        self.assertTrue((self.snapshots_dir / f"{snap['snapshot_id']}.json").exists())
        self.assertFalse((self.iu_xray_dir / f"{snap['snapshot_id']}.json").exists())


class TestExperimentManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir) / "experiments"
        self.snapshots_dir = Path(self.temp_dir) / "snapshots"
        self.iu_xray_dir = Path(self.temp_dir) / "iu_xray"

        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.iu_xray_dir.mkdir(parents=True, exist_ok=True)

        (self.iu_xray_dir / "CXR1122").mkdir(parents=True, exist_ok=True)
        with open(self.iu_xray_dir / "CXR1122" / "evidence_findings.json", "w") as f:
            json.dump({"study_id": "CXR1122"}, f)

        self.snapshot_mgr = DatasetSnapshotManager(
            snapshots_dir=str(self.snapshots_dir),
            data_dir=str(self.iu_xray_dir)
        )
        self.test_snap = self.snapshot_mgr.create_snapshot(
            name="Benchmark Snapshot",
            description="Testing benchmark",
            study_ids=["CXR1122"]
        )

        self.exp_mgr = ExperimentManager(
            experiments_dir=str(self.experiments_dir),
            snapshot_manager=self.snapshot_mgr
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_configuration_fingerprint_determinism(self):
        cfg = {
            "model_name": "TorchXRayVision DenseNet-121",
            "model_version": "densenet121-res224-all",
            "target_layer": "model.features.norm5",
            "qa_threshold": 0.15,
            "qa_top_k": 5
        }
        fp1 = self.exp_mgr.generate_configuration_fingerprint(cfg)
        fp2 = self.exp_mgr.generate_configuration_fingerprint(cfg)
        self.assertEqual(fp1["fingerprint_sha256"], fp2["fingerprint_sha256"])
        self.assertEqual(len(fp1["fingerprint_sha256"]), 64)

    def test_configuration_fingerprint_excludes_keys_and_secrets(self):
        cfg = {
            "model_name": "DenseNet-121",
            "api_key": "SECRET_GEMINI_KEY_DO_NOT_LEAK",
            "password": "admin_password",
            "qa_threshold": 0.20
        }
        fp = self.exp_mgr.generate_configuration_fingerprint(cfg)
        fp_json = json.dumps(fp)
        self.assertNotIn("SECRET_GEMINI_KEY", fp_json)
        self.assertNotIn("admin_password", fp_json)

    def test_create_experiment_lifecycle(self):
        exp = self.exp_mgr.create_experiment(
            name="Baseline Experiment",
            description="Testing DenseNet-121 norm5 baseline",
            dataset_snapshot_id=self.test_snap["snapshot_id"],
            configuration={"model_name": "DenseNet-121", "target_layer": "model.features.norm5"},
            created_by="researcher_01"
        )
        self.assertEqual(exp["status"], "CREATED")
        self.assertEqual(exp["dataset_snapshot_id"], self.test_snap["snapshot_id"])
        self.assertIn("configuration_fingerprint", exp)
        self.assertEqual(exp["created_by"], "researcher_01")

        # Verify disk persistence in metadata.json
        exp_dir = self.experiments_dir / exp["experiment_id"]
        self.assertTrue((exp_dir / "metadata.json").exists())

    def test_experiment_state_transitions(self):
        exp = self.exp_mgr.create_experiment(
            name="Lifecycle Test",
            description="Testing states",
            dataset_snapshot_id=self.test_snap["snapshot_id"]
        )
        exp_id = exp["experiment_id"]

        # 1. Update to RUNNING
        updated = self.exp_mgr.update_experiment_status(exp_id, "RUNNING")
        self.assertEqual(updated["status"], "RUNNING")

        # 2. Complete experiment with metrics
        completed = self.exp_mgr.update_experiment_status(
            experiment_id=exp_id,
            new_status="COMPLETED",
            metrics={"review_coverage": {"review_completion_ratio": 1.0}}
        )
        self.assertEqual(completed["status"], "COMPLETED")
        self.assertIsNotNone(completed["completed_at"])

    def test_experiment_immutability_on_completed(self):
        exp = self.exp_mgr.create_experiment(
            name="Immutability Test",
            description="Testing completed lock",
            dataset_snapshot_id=self.test_snap["snapshot_id"]
        )
        exp_id = exp["experiment_id"]
        self.exp_mgr.update_experiment_status(exp_id, "RUNNING")
        self.exp_mgr.update_experiment_status(
            experiment_id=exp_id,
            new_status="COMPLETED",
            metrics={"review_coverage": {"review_completion_ratio": 1.0}}
        )

        # Attempting to transition completed experiment back to RUNNING or CREATED must fail
        with self.assertRaises(ValueError):
            self.exp_mgr.update_experiment_status(
                experiment_id=exp_id,
                new_status="RUNNING"
            )

    def test_experiment_archive_lifecycle(self):
        exp = self.exp_mgr.create_experiment(
            name="Archival Test",
            description="Testing archival",
            dataset_snapshot_id=self.test_snap["snapshot_id"]
        )
        exp_id = exp["experiment_id"]
        self.exp_mgr.update_experiment_status(exp_id, "RUNNING")
        self.exp_mgr.update_experiment_status(exp_id, "COMPLETED")

        archived = self.exp_mgr.archive_experiment(exp_id, reason="Completed research run archived")
        self.assertEqual(archived["status"], "ARCHIVED")
        self.assertEqual(archived["archive_reason"], "Completed research run archived")

    def test_experiment_id_path_traversal_protection(self):
        with self.assertRaises(ValueError):
            self.exp_mgr._sanitize_id("../../etc/passwd")

    def test_invalid_snapshot_id_rejected(self):
        with self.assertRaises(ValueError):
            self.exp_mgr.create_experiment(
                name="Bad Snapshot",
                description="Should fail",
                dataset_snapshot_id="NON_EXISTENT_SNAPSHOT_ID"
            )


class TestResearchMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = ResearchMetricsCalculator()

    def test_calculate_metrics_for_studies(self):
        metrics = self.calc.calculate_metrics_for_studies(["CXR1122"])
        self.assertIn("review_coverage", metrics)
        self.assertIn("consensus_coverage", metrics)
        self.assertIn("machine_reviewer_agreement", metrics)
        self.assertIn("inter_rater_reliability", metrics)
        self.assertIn("explainability_coverage", metrics)
        self.assertIn("human_corrections", metrics)

    def test_calculate_metrics_kappa_bounds(self):
        metrics = self.calc.calculate_metrics_for_studies(["CXR1122"])
        irr = metrics["inter_rater_reliability"]
        ck = irr.get("cohens_kappa_2_reviewers", {}).get("average_kappa")
        fk = irr.get("fleiss_kappa_multi_reviewers", {}).get("average_kappa")
        if ck is not None:
            self.assertGreaterEqual(ck, -1.0)
            self.assertLessEqual(ck, 1.0)
        if fk is not None:
            self.assertGreaterEqual(fk, -1.0)
            self.assertLessEqual(fk, 1.0)

    def test_calculate_metrics_no_nan_or_infinity(self):
        metrics = self.calc.calculate_metrics_for_studies([])
        metrics_json = json.dumps(metrics)
        self.assertNotIn("NaN", metrics_json)
        self.assertNotIn("Infinity", metrics_json)
        self.assertNotIn("-Infinity", metrics_json)


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir) / "experiments"
        self.snapshots_dir = Path(self.temp_dir) / "snapshots"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_mgr = DatasetSnapshotManager(snapshots_dir=str(self.snapshots_dir))
        self.test_snap = self.snapshot_mgr.create_snapshot(
            name="Runner Benchmark Snap",
            description="Testing runner",
            study_ids=["CXR1122"]
        )

        self.exp_mgr = ExperimentManager(
            experiments_dir=str(self.experiments_dir),
            snapshot_manager=self.snapshot_mgr
        )
        self.runner = ExperimentRunner(
            experiment_manager=self.exp_mgr,
            snapshot_manager=self.snapshot_mgr,
            experiments_dir=str(self.experiments_dir)
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_experiment_end_to_end_and_provenance(self):
        exp = self.exp_mgr.create_experiment(
            name="E2E Runner Test",
            description="Testing 11 stages",
            dataset_snapshot_id=self.test_snap["snapshot_id"],
            configuration={"model_name": "DenseNet-121", "target_layer": "model.features.norm5"}
        )
        exp_id = exp["experiment_id"]

        result = self.runner.run_experiment(exp_id)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIn("metrics", result)
        self.assertIn("provenance", result)

        # Verify 11-stage provenance pipeline
        stages = result["provenance"]["pipeline_stages"]
        self.assertEqual(len(stages), 11)
        for i, stage in enumerate(stages, start=1):
            self.assertEqual(stage["status"], "COMPLETED")

    def test_run_experiment_cannot_mutate_iu_xray_data(self):
        evidence_file = BACKEND_DIR.parent / "data" / "iu_xray" / "CXR1122" / "evidence_findings.json"
        if evidence_file.exists():
            import hashlib
            with open(evidence_file, "rb") as f:
                hash_before = hashlib.sha256(f.read()).hexdigest()

            exp = self.exp_mgr.create_experiment(
                name="Immutability Check",
                description="Checking data/iu_xray byte immutability",
                dataset_snapshot_id=self.test_snap["snapshot_id"]
            )
            self.runner.run_experiment(exp["experiment_id"])

            with open(evidence_file, "rb") as f:
                hash_after = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(hash_before, hash_after)

    def test_export_experiment_json_and_text(self):
        exp = self.exp_mgr.create_experiment(
            name="Export Test",
            description="Testing exports",
            dataset_snapshot_id=self.test_snap["snapshot_id"]
        )
        self.runner.run_experiment(exp["experiment_id"])

        # JSON Export
        json_txt, mime = self.runner.export_experiment(exp["experiment_id"], format="json")
        self.assertEqual(mime, "application/json")
        pkg = json.loads(json_txt)
        self.assertIn("experiment_id", pkg)
        self.assertIn("research_disclaimer", pkg)
        self.assertIn("RESEARCH EVALUATION ONLY", pkg["research_disclaimer"])

        # Text Export
        txt, mime = self.runner.export_experiment(exp["experiment_id"], format="text")
        self.assertTrue(mime.startswith("text/plain"))
        self.assertIn("RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE", txt)
        self.assertIn(exp["experiment_id"], txt)


class TestExperimentComparator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir) / "experiments"
        self.snapshots_dir = Path(self.temp_dir) / "snapshots"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_mgr = DatasetSnapshotManager(snapshots_dir=str(self.snapshots_dir))
        self.exp_mgr = ExperimentManager(
            experiments_dir=str(self.experiments_dir),
            snapshot_manager=self.snapshot_mgr
        )
        self.comparator = ExperimentComparator(experiment_manager=self.exp_mgr)

        snap = self.snapshot_mgr.create_snapshot(name="Snap 1", description="Snap", study_ids=["CXR1122"])

        # Create 2 experiments with different configurations
        self.exp1 = self.exp_mgr.create_experiment(
            name="Exp Layer Norm5",
            description="Exp 1",
            dataset_snapshot_id=snap["snapshot_id"],
            configuration={"target_layer": "model.features.norm5", "qa_threshold": 0.15}
        )
        self.exp_mgr.update_experiment_status(self.exp1["experiment_id"], "RUNNING")
        self.exp_mgr.update_experiment_status(
            self.exp1["experiment_id"],
            new_status="COMPLETED",
            metrics={"review_coverage": {"review_completion_ratio": 1.0}, "machine_reviewer_agreement": {"overall_agreement_ratio": 0.80}}
        )

        self.exp2 = self.exp_mgr.create_experiment(
            name="Exp Layer DenseBlock4",
            description="Exp 2",
            dataset_snapshot_id=snap["snapshot_id"],
            configuration={"target_layer": "model.features.denseblock4", "qa_threshold": 0.25}
        )
        self.exp_mgr.update_experiment_status(self.exp2["experiment_id"], "RUNNING")
        self.exp_mgr.update_experiment_status(
            self.exp2["experiment_id"],
            new_status="COMPLETED",
            metrics={"review_coverage": {"review_completion_ratio": 1.0}, "machine_reviewer_agreement": {"overall_agreement_ratio": 0.75}}
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compare_experiments_config_differences(self):
        comp = self.comparator.compare_experiments([self.exp1["experiment_id"], self.exp2["experiment_id"]])
        self.assertEqual(len(comp["experiment_ids"]), 2)
        diffs = {d["configuration_field"]: d for d in comp["configuration_differences"]}
        self.assertTrue(diffs["target_layer"]["has_difference"])
        self.assertTrue(diffs["qa_threshold"]["has_difference"])
        self.assertNotIn("model_name", diffs)  # identical parameters omitted from differing list

    def test_compare_experiments_metric_comparison(self):
        comp = self.comparator.compare_experiments([self.exp1["experiment_id"], self.exp2["experiment_id"]])
        m_comp = comp["metric_comparisons"]
        self.assertIn("machine_reviewer_agreement", m_comp)
        self.assertEqual(m_comp["machine_reviewer_agreement"][self.exp1["experiment_id"]]["overall_agreement_ratio"], 0.80)
        self.assertEqual(m_comp["machine_reviewer_agreement"][self.exp2["experiment_id"]]["overall_agreement_ratio"], 0.75)

    def test_compare_experiments_non_evaluative_terminology(self):
        comp = self.comparator.compare_experiments([self.exp1["experiment_id"], self.exp2["experiment_id"]])
        comp_str = json.dumps(comp).lower()
        # Non-evaluative invariant: Zero ranking/grading language
        self.assertNotIn("winner", comp_str)
        self.assertNotIn("superior", comp_str)
        self.assertNotIn("best", comp_str)
        self.assertNotIn("rank", comp_str)
        self.assertNotIn("grade", comp_str)

    def test_compare_experiments_insufficient_ids_rejection(self):
        with self.assertRaises(ValueError):
            self.comparator.compare_experiments([self.exp1["experiment_id"]])


class TestExperimentValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir) / "experiments"
        self.snapshots_dir = Path(self.temp_dir) / "snapshots"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_mgr = DatasetSnapshotManager(snapshots_dir=str(self.snapshots_dir))
        self.exp_mgr = ExperimentManager(
            experiments_dir=str(self.experiments_dir),
            snapshot_manager=self.snapshot_mgr
        )
        self.validator = ExperimentValidator(
            experiment_manager=self.exp_mgr,
            snapshot_manager=self.snapshot_mgr
        )

        snap = self.snapshot_mgr.create_snapshot(name="Val Snap", description="Val", study_ids=["CXR1122"])

        self.exp = self.exp_mgr.create_experiment(
            name="Valid Exp",
            description="Validator test",
            dataset_snapshot_id=snap["snapshot_id"],
            configuration={"model_name": "TorchXRayVision DenseNet-121", "target_layer": "model.features.norm5", "qa_threshold": 0.15, "qa_top_k": 5}
        )
        self.exp_mgr.update_experiment_status(self.exp["experiment_id"], "RUNNING")
        self.exp_mgr.update_experiment_status(
            self.exp["experiment_id"],
            new_status="COMPLETED",
            metrics={
                "review_coverage": {"review_completion_ratio": 0.8},
                "inter_rater_reliability": {
                    "cohens_kappa_2_reviewers": {"average_kappa": 0.65},
                    "fleiss_kappa_multi_reviewers": {"average_kappa": 0.60}
                }
            },
            provenance={"pipeline_stages": [{"stage_id": f"s_{i}", "stage_name": f"Stage {i}", "status": "COMPLETED"} for i in range(11)]}
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_experiment_all_invariants_pass(self):
        report = self.validator.validate(self.exp["experiment_id"])
        self.assertTrue(report["is_valid"], msg=f"Violations found: {report.get('violations')}")
        self.assertEqual(len(report["violations"]), 0)

    def test_validate_experiment_detects_fingerprint_mismatch(self):
        exp_file = self.experiments_dir / self.exp["experiment_id"] / "metadata.json"
        with open(exp_file, "r") as f:
            data = json.load(f)
        data["configuration_fingerprint"]["fingerprint_sha256"] = "corrupted_hash_value"
        with open(exp_file, "w") as f:
            json.dump(data, f)

        report = self.validator.validate(self.exp["experiment_id"])
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("fingerprint mismatch" in v.lower() for v in report["violations"]))

    def test_validate_experiment_detects_metric_out_of_bounds(self):
        exp_file = self.experiments_dir / self.exp["experiment_id"] / "metadata.json"
        with open(exp_file, "r") as f:
            data = json.load(f)
        # Ratio cannot exceed 1.0
        data["metrics"]["review_coverage"]["review_completion_ratio"] = 1.5
        with open(exp_file, "w") as f:
            json.dump(data, f)

        report = self.validator.validate(self.exp["experiment_id"])
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("out of bounds" in v.lower() for v in report["violations"]))

    def test_validate_experiment_detects_leaked_ground_truth(self):
        exp_file = self.experiments_dir / self.exp["experiment_id"] / "metadata.json"
        with open(exp_file, "r") as f:
            data = json.load(f)
        data["description"] = "Test run with reference report leak: <eFind> infiltration</eFind>"
        with open(exp_file, "w") as f:
            json.dump(data, f)

        report = self.validator.validate(self.exp["experiment_id"])
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("ground-truth" in v.lower() for v in report["violations"]))


class TestPhase15ApiEndpoints(unittest.TestCase):
    """Integration tests for Phase 1.5 REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        import threading
        import time
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        from api import create_server

        cls.server_port = 8955
        cls.server = create_server(port=cls.server_port, host="127.0.0.1")
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _api_get(self, endpoint: str):
        from urllib.request import Request, urlopen
        url = f"http://127.0.0.1:{self.server_port}{endpoint}"
        req = Request(url, method="GET", headers={"Connection": "close"})
        with urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in content_type:
                return json.loads(raw.decode("utf-8")), resp.status
            return raw.decode("utf-8"), resp.status

    def _api_post(self, endpoint: str, payload: dict):
        from urllib.request import Request, urlopen
        url = f"http://127.0.0.1:{self.server_port}{endpoint}"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data_bytes, method="POST", headers={"Content-Type": "application/json", "Connection": "close"})
        with urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")), resp.status

    def test_api_snapshots_list_and_create(self):
        # 1. Create Snapshot via POST
        snap_payload = {
            "name": "API Test Snapshot",
            "description": "Created via API test",
            "study_ids": ["CXR1122"],
            "created_by": "api_test_curator"
        }
        res, status = self._api_post("/api/experiments/snapshots", snap_payload)
        self.assertEqual(status, 201)
        self.assertIn("snapshot_id", res)
        snap_id = res["snapshot_id"]

        # 2. List Snapshots via GET
        list_res, list_status = self._api_get("/api/experiments/snapshots")
        self.assertEqual(list_status, 200)
        self.assertIn("snapshots", list_res)
        self.assertTrue(any(s["snapshot_id"] == snap_id for s in list_res["snapshots"]))

        # 3. Get Single Snapshot via GET
        item_res, item_status = self._api_get(f"/api/experiments/snapshots/{snap_id}")
        self.assertEqual(item_status, 200)
        self.assertEqual(item_res["snapshot_id"], snap_id)

    def test_api_experiments_crud_and_runner(self):
        # 1. Ensure snapshot exists
        snap_res, _ = self._api_post("/api/experiments/snapshots", {
            "name": "Exp API Snap",
            "study_ids": ["CXR1122"]
        })
        snap_id = snap_res["snapshot_id"]

        # 2. Create Experiment via POST
        exp_payload = {
            "name": "API E2E Experiment",
            "description": "Integration test experiment",
            "snapshot_id": snap_id,
            "model_name": "TorchXRayVision DenseNet-121",
            "model_version": "densenet121-res224-all",
            "target_layer": "model.features.norm5",
            "qa_threshold": 0.15,
            "qa_top_k": 5,
            "created_by": "api_tester"
        }
        exp_res, exp_status = self._api_post("/api/experiments", exp_payload)
        self.assertEqual(exp_status, 201)
        self.assertIn("experiment_id", exp_res)
        exp_id = exp_res["experiment_id"]

        # 3. List Experiments via GET
        list_res, list_status = self._api_get("/api/experiments")
        self.assertEqual(list_status, 200)
        self.assertIn("experiments", list_res)
        self.assertTrue(any(e["experiment_id"] == exp_id for e in list_res["experiments"]))

        # 4. Run Experiment via POST
        run_res, run_status = self._api_post(f"/api/experiments/{exp_id}/run", {"executed_by": "api_tester"})
        self.assertEqual(run_status, 200)
        self.assertTrue(run_res.get("success", False))

        # 5. Fetch Details, Metrics, Provenance via GET
        det_res, det_status = self._api_get(f"/api/experiments/{exp_id}")
        self.assertEqual(det_status, 200)
        self.assertEqual(det_res["status"], "COMPLETED")
        self.assertIn("metrics", det_res)

        prov_res, prov_status = self._api_get(f"/api/experiments/{exp_id}/provenance")
        self.assertEqual(prov_status, 200)
        self.assertIn("pipeline_stages", prov_res)
        self.assertEqual(len(prov_res["pipeline_stages"]), 11)

        # 6. Validate Experiment via GET
        val_res, val_status = self._api_get(f"/api/experiments/{exp_id}/validation")
        self.assertEqual(val_status, 200)
        self.assertTrue(val_res.get("is_valid", False))

        # 7. Export JSON and Text via GET
        exp_json, ej_status = self._api_get(f"/api/experiments/{exp_id}/export?format=json")
        self.assertEqual(ej_status, 200)
        self.assertIn("experiment_id", exp_json)

        exp_txt, et_status = self._api_get(f"/api/experiments/{exp_id}/export?format=text")
        self.assertEqual(et_status, 200)
        self.assertIn("RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE", exp_txt)

        # 8. Archive Experiment via POST
        arch_res, arch_status = self._api_post(f"/api/experiments/{exp_id}/archive", {"reason": "Test complete"})
        self.assertEqual(arch_status, 200)
        self.assertTrue(arch_res.get("success", False))

    def test_api_compare_experiments(self):
        # Create snap and two experiments
        snap_res, _ = self._api_post("/api/experiments/snapshots", {"name": "Compare Snap", "study_ids": ["CXR1122"]})
        snap_id = snap_res["snapshot_id"]

        e1_res, _ = self._api_post("/api/experiments", {"name": "Exp 1", "snapshot_id": snap_id, "target_layer": "norm5"})
        e2_res, _ = self._api_post("/api/experiments", {"name": "Exp 2", "snapshot_id": snap_id, "target_layer": "denseblock4"})

        e1_id, e2_id = e1_res["experiment_id"], e2_res["experiment_id"]
        self._api_post(f"/api/experiments/{e1_id}/run", {})
        self._api_post(f"/api/experiments/{e2_id}/run", {})

        # Compare
        comp_res, comp_status = self._api_post("/api/experiments/compare", {"experiment_ids": [e1_id, e2_id]})
        self.assertEqual(comp_status, 200)
        self.assertIn("configuration_differences", comp_res)
        self.assertIn("metric_comparisons", comp_res)
        self.assertIn("research_disclaimer", comp_res)


if __name__ == "__main__":
    unittest.main()

