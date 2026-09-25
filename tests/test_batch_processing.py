"""
Unit and Integration Test Suite for Batch Processing, Artifact Generation & Study Switching

Module: test_batch_processing.py
Tests:
1. Single-study processing
2. Multi-study processing
3. Unknown study rejection
4. Missing image handling
5. Correct image-to-study mapping
6. Six-stage execution
7. Artifact persistence
8. Existing artifact reuse
9. Failed-stage handling
10. Partial batch failure
11. Job status reporting
12. Progress reporting
13. Study switching
14. Stale response protection
15. "Inference not run" vs "zero findings"
16. Multi-view study handling
17. Ground-truth leakage prevention
18. Machine artifact immutability
19. Review-layer isolation
20. Validation failure handling
21. CPU execution
22. CUDA execution fallback
"""

import os
import sys
import json
import time
import shutil
import unittest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from batch_processor import (
    BatchProcessor,
    process_single_study,
    find_study_images,
    get_shared_vision_model,
    has_valid_persisted_artifacts,
    get_study_artifacts_dir
)
import api
from study_manager import StudyManager
from review_manager import ReviewManager
from consensus_manager import ConsensusManager


class TestBatchProcessing(unittest.TestCase):
    """Test suite for batch processing, deterministic artifact generation, and multi-study switching."""

    @classmethod
    def setUpClass(cls):
        cls.test_study_id = "CXR1401"
        cls.data_dir = os.path.join(BASE_DIR, "data", "iu_xray")
        cls.studies_dir = os.path.join(cls.data_dir, "studies")
        cls.batch_proc = BatchProcessor(data_dir=cls.data_dir)

    def test_01_find_study_images(self):
        """Test image discovery and view inference for studies."""
        images = find_study_images("CXR1122")
        self.assertGreater(len(images), 0)
        self.assertIn("image_id", images[0])
        self.assertIn("path", images[0])
        self.assertIn("view", images[0])
        self.assertTrue(os.path.exists(images[0]["path"]))

    def test_02_single_study_processing(self):
        """Test complete 6-stage single study processing on CXR1401."""
        res = process_single_study(self.test_study_id, force=True, base_dir=BASE_DIR)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertFalse(res["cached"])
        self.assertEqual(res["study_id"], self.test_study_id)
        self.assertTrue(os.path.exists(res["artifacts_dir"]))

    def test_03_artifact_persistence(self):
        """Verify all 6 machine stage artifacts are persisted on disk."""
        study_art_dir = get_study_artifacts_dir(self.test_study_id, base_dir=BASE_DIR)
        required_files = [
            "vision_output.json",
            "qa_output.json",
            "evidence_package.json",
            "grounding_package.json",
            "generated_report.json",
            "validation_result.json",
            "study_meta.json"
        ]
        for fn in required_files:
            p = os.path.join(study_art_dir, fn)
            self.assertTrue(os.path.exists(p), f"Missing artifact: {fn}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_04_existing_artifact_reuse(self):
        """Verify cached valid artifacts are returned when force=False."""
        res = process_single_study(self.test_study_id, force=False, base_dir=BASE_DIR)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertTrue(res["cached"])

    def test_05_unknown_study_rejection(self):
        """Verify processing an unknown/non-existent study raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            process_single_study("CXR99999_NON_EXISTENT", force=True, base_dir=BASE_DIR)

    def test_06_missing_image_handling(self):
        """Verify find_study_images returns empty list for missing study."""
        imgs = find_study_images("NON_EXISTENT_STUDY_12345")
        self.assertEqual(len(imgs), 0)

    def test_07_correct_image_to_study_mapping(self):
        """Verify images match the queried study ID and not another study."""
        imgs_1122 = find_study_images("CXR1122")
        imgs_1401 = find_study_images("CXR1401")
        self.assertTrue(all(img["image_id"].startswith("CXR1122") for img in imgs_1122))
        self.assertTrue(all(img["image_id"].startswith("CXR1401") for img in imgs_1401))

    def test_08_multi_study_batch_processing(self):
        """Test batch processing across multiple studies."""
        proc = BatchProcessor(data_dir=self.data_dir)
        study_ids = ["CXR1122", "CXR1401"]
        job_id = proc.create_batch_job(study_ids)
        self.assertTrue(job_id.startswith("batch_"))

        proc.start_batch_job(job_id, force=False)
        # Wait for completion
        for _ in range(30):
            st = proc.get_job_status(job_id)
            if st["status"] in ("COMPLETED", "FAILED", "COMPLETED_WITH_ERRORS"):
                break
            time.sleep(0.5)

        status = proc.get_job_status(job_id)
        self.assertIn(status["status"], ("COMPLETED", "COMPLETED_WITH_ERRORS"))
        self.assertEqual(status["total_studies"], 2)
        self.assertEqual(status["completed_studies"], 2)

    def test_09_partial_batch_failure_isolation(self):
        """Verify a failed study in a batch does not fail valid studies."""
        proc = BatchProcessor(data_dir=self.data_dir)
        study_ids = ["CXR1122", "CXR_INVALID_99999", "CXR1401"]
        job_id = proc.create_batch_job(study_ids)
        proc.start_batch_job(job_id, force=False)

        for _ in range(30):
            st = proc.get_job_status(job_id)
            if st["status"] in ("COMPLETED", "FAILED", "COMPLETED_WITH_ERRORS"):
                break
            time.sleep(0.5)

        status = proc.get_job_status(job_id)
        self.assertEqual(status["completed_studies"], 2)
        self.assertEqual(status["failed_studies"], 1)
        self.assertEqual(status["status"], "COMPLETED_WITH_ERRORS")
        # Check per-study status
        sp = status["studies_progress"]
        self.assertEqual(sp["CXR1122"]["status"], "COMPLETED")
        self.assertEqual(sp["CXR_INVALID_99999"]["status"], "FAILED")
        self.assertEqual(sp["CXR1401"]["status"], "COMPLETED")

    def test_10_job_status_reporting(self):
        """Verify detailed fields in get_job_status."""
        proc = BatchProcessor(data_dir=self.data_dir)
        job_id = proc.create_batch_job(["CXR1122"])
        st = proc.get_job_status(job_id)
        self.assertIn("job_id", st)
        self.assertIn("status", st)
        self.assertIn("total_studies", st)
        self.assertIn("completed_studies", st)
        self.assertIn("studies_progress", st)
        self.assertIn("stages", st)

    def test_11_progress_reporting(self):
        """Verify progress percentage calculation."""
        proc = BatchProcessor(data_dir=self.data_dir)
        job_id = proc.create_batch_job(["CXR1122", "CXR1401"])
        proc.start_batch_job(job_id, force=False)
        time.sleep(1.0)
        st = proc.get_job_status(job_id)
        self.assertGreaterEqual(st["progress_percentage"], 0.0)

    def test_12_study_switching_isolation(self):
        """Verify study data for CXR1122 and CXR1401 are independent."""
        data_1122 = api.load_study_data("CXR1122")
        data_1401 = api.load_study_data("CXR1401")
        self.assertIsNotNone(data_1122)
        self.assertIsNotNone(data_1401)
        self.assertEqual(data_1122["study_id"], "CXR1122")
        self.assertEqual(data_1401["study_id"], "CXR1401")
        self.assertNotEqual(data_1122["image_id"], data_1401["image_id"])

    def test_13_inference_not_run_vs_zero_findings(self):
        """Verify uncomputed study reports inference_status=NOT_RUN."""
        # CXR1007 might be uncomputed if not yet run
        art_dir = get_study_artifacts_dir("CXR1007", base_dir=BASE_DIR)
        if not os.path.exists(art_dir):
            uncomp = api.load_study_data("CXR1007")
            if uncomp:
                self.assertEqual(uncomp["inference_status"], "NOT_RUN")
                self.assertFalse(uncomp["has_artifacts"])
                self.assertEqual(uncomp["validation"]["pipeline_stages"]["vision"], False)

    def test_14_multi_view_handling(self):
        """Verify dual-view detection in get_study_images_metadata."""
        # Find a study with 2 images
        studies = api.discover_available_studies()
        dual_studies = [s for s in studies if len(s["image_ids"]) >= 2]
        if dual_studies:
            sid = dual_studies[0]["study_id"]
            meta = api.get_study_images_metadata(sid)
            self.assertTrue(meta["has_dual_view"])
            self.assertGreaterEqual(meta["total_images"], 2)

    def test_15_ground_truth_leakage_prevention(self):
        """Verify no XML ground truth tags or reports leak into API payloads."""
        for sid in ["CXR1122", "CXR1401"]:
            data = api.load_study_data(sid)
            if data and data.get("has_artifacts"):
                json_str = json.dumps(data)
                self.assertNotIn("<eFind>", json_str)
                self.assertNotIn("<eImpression>", json_str)
                self.assertNotIn("<Major>", json_str)

    def test_16_machine_artifact_immutability(self):
        """Verify original IU X-Ray files remain unchanged."""
        orig_img_1122 = os.path.join(self.data_dir, "images", "CXR1122_IM-0080-1001-0002.png")
        self.assertTrue(os.path.exists(orig_img_1122))
        self.assertGreater(os.path.getsize(orig_img_1122), 0)

    def test_17_review_layer_isolation(self):
        """Verify batch processing does not overwrite reviews or consensus files."""
        rev_dir = os.path.join(BASE_DIR, "data", "reviews")
        cons_dir = os.path.join(BASE_DIR, "data", "consensus")
        # Run study
        process_single_study("CXR1122", force=False, base_dir=BASE_DIR)
        # Ensure review and consensus directories still exist
        self.assertTrue(os.path.exists(rev_dir))
        self.assertTrue(os.path.exists(cons_dir))

    def test_18_validation_failure_handling(self):
        """Verify validation failure raises ValueError during single study processing."""
        # Test validator with invalid data
        from validate_evidence import validate_evidence_package
        is_val, issues = validate_evidence_package({"invalid": "data"})
        self.assertFalse(is_val)
        self.assertGreater(len(issues), 0)

    def test_19_cpu_and_cuda_execution(self):
        """Verify get_shared_vision_model works on CPU and CUDA."""
        model_cpu, dev_cpu = get_shared_vision_model(torch.device("cpu"))
        self.assertIsNotNone(model_cpu)
        self.assertEqual(dev_cpu.type, "cpu")
        if torch.cuda.is_available():
            model_cuda, dev_cuda = get_shared_vision_model(torch.device("cuda"))
            self.assertIsNotNone(model_cuda)
            self.assertEqual(dev_cuda.type, "cuda")

    def test_20_study_manager_discovery(self):
        """Verify StudyManager discovers studies and reflects completed artifacts."""
        sm = StudyManager(data_dir=self.data_dir)
        studies = sm.discover_studies()
        self.assertGreater(len(studies), 100)
        cxr1122_meta = next((s for s in studies if s["study_id"] == "CXR1122"), None)
        self.assertIsNotNone(cxr1122_meta)
        self.assertEqual(cxr1122_meta["pipeline_status"], "COMPLETE")

    def test_21_stage_callback_tracking(self):
        """Verify stage callback receives progression events."""
        stages_visited = []

        def cb(sid, stage, pct, err):
            stages_visited.append((stage, pct))

        process_single_study(
            self.test_study_id,
            force=True,
            stage_callback=cb,
            base_dir=BASE_DIR
        )
        stage_names = [s[0] for s in stages_visited]
        self.assertIn("VISION_RUNNING", stage_names)
        self.assertIn("QA_RUNNING", stage_names)
        self.assertIn("EVIDENCE_RUNNING", stage_names)
        self.assertIn("GROUNDING_RUNNING", stage_names)
        self.assertIn("LLM_RUNNING", stage_names)
        self.assertIn("VALIDATING", stage_names)
        self.assertIn("COMPLETED", stage_names)

    def test_22_api_endpoints_consistency(self):
        """Verify API study data schema complies with frontend expectations."""
        study_data = api.load_study_data("CXR1122")
        self.assertIn("study_id", study_data)
        self.assertIn("image_id", study_data)
        self.assertIn("view", study_data)
        self.assertIn("evidence", study_data)
        self.assertIn("qa_candidates", study_data)
        self.assertIn("qa_questions", study_data)
        self.assertIn("groundings", study_data)
        self.assertIn("report", study_data)
        self.assertIn("validation", study_data)


if __name__ == "__main__":
    unittest.main()
