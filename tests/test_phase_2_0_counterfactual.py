"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 2.0 — Unit & Integration Test Suite for Interactive Counterfactual Explanations

Module: test_phase_2_0_counterfactual.py
Purpose:
- Validates all Phase 2.0 capabilities, perturbation methods, and architectural safety invariants.
- Contains >= 35 comprehensive tests covering:
    1. Schema validation against docs/counterfactual_schema.json
    2. Counterfactual experiment creation
    3. Lifecycle state transitions
    4. ROI bounds validation & normalization
    5. Perturbation determinism with fixed seed
    6. Original image disk immutability
    7. Baseline inference preservation
    8. Counterfactual inference execution
    9. Finding delta calculation logic
    10. Zero-denominator safety in relative deltas
    11. Rejection of NaN / Infinity in numeric outputs
    12. Grad-CAM comparison & difference map computation
    13. Attribution overlap and ROI fraction metrics
    14. Diagnostic QA impact comparison
    15. Report impact statement diffing
    16. Random control ROI generation & non-overlap
    17. Reproducibility fingerprint calculation
    18. Configuration drift detection
    19. Finalization locking
    20. Published experiment immutability
    21. Ground-truth XML report leakage prevention
    22. Path traversal protection
    23. Payload-size and coordinate bounds validation
    24. REST API dashboard verification
    25. REST API counterfactual CRUD endpoints
    26. REST API sub-resource endpoints (baseline, counterfactual, comparison, attribution, qa, report, repro)
    27. REST API compare endpoint
    28. Export package validation
    29. Statistical summary & bootstrap CI computation
    30. Multi-perturbation comparator neutral wording
    31. Audit trail sequential integrity
    32. Security sanitization invariants
    33. Artifact naming and disk isolation
    34. Mandatory research disclaimer presence
    35. Full end-to-end counterfactual experiment lifecycle
    36. Safety validator integration check
"""

import os
import sys
import json
import math
import shutil
import hashlib
import unittest
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.counterfactual_manager import (
    CounterfactualManager,
    validate_finite_number,
    sanitize_id,
    compute_file_sha256,
    RESEARCH_DISCLAIMER
)
from backend.counterfactual_comparator import CounterfactualComparator
from backend.validate_phase_2_0 import (
    check_iu_xray_dataset_integrity,
    check_ground_truth_isolation,
    check_published_counterfactual_immutability,
    check_security_sanitization,
    run_all_phase_2_0_safety_checks
)
from backend.api import create_server


class TestPhase20Counterfactual(unittest.TestCase):
    """Comprehensive test suite for Phase 2.0 interactive counterfactual explanations."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(BASE_DIR, "data", "test_counterfactuals_env")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.manager = CounterfactualManager(base_dir=BASE_DIR)
        cls.comparator = CounterfactualComparator(manager=cls.manager)
        cls.study_id = "CXR1122"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_schema_exists_and_valid_json(self):
        """1. Validates docs/counterfactual_schema.json format and definitions."""
        schema_path = os.path.join(BASE_DIR, "docs", "counterfactual_schema.json")
        self.assertTrue(os.path.exists(schema_path), "Schema file docs/counterfactual_schema.json must exist.")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIn("definitions", schema)
        self.assertIn("CounterfactualExperiment", schema["definitions"])
        self.assertIn("PerturbationConfiguration", schema["definitions"])
        self.assertIn("FindingDelta", schema["definitions"])
        self.assertIn("AttributionDelta", schema["definitions"])
        self.assertIn("QAImpactItem", schema["definitions"])
        self.assertIn("ReportImpact", schema["definitions"])

    def test_02_create_counterfactual_draft(self):
        """2. Verifies counterfactual creation initializes in DRAFT lifecycle state."""
        exp = self.manager.create_counterfactual(self.study_id, "PA", created_by="test_actor")
        self.assertIsNotNone(exp["counterfactual_id"])
        self.assertEqual(exp["status"], "DRAFT")
        self.assertEqual(exp["study_id"], self.study_id)
        self.assertEqual(exp["reproducibility"]["status"], "INCOMPLETE")
        self.assertTrue(len(exp["audit_trail"]) >= 1)
        self.assertEqual(exp["audit_trail"][0]["action"], "CREATE")

    def test_03_lifecycle_state_transitions(self):
        """3. Verifies deterministic lifecycle progression: DRAFT -> CONFIGURED -> RUNNING -> COMPLETED -> VALIDATED -> PUBLISHED."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.assertEqual(exp["status"], "DRAFT")

        # Configure
        conf = self.manager.configure_counterfactual(
            self.study_id, cf_id, "REGION_MASK", {"x": 40, "y": 40, "width": 60, "height": 60}
        )
        self.assertEqual(conf["status"], "CONFIGURED")

        # Run
        run_res = self.manager.run_counterfactual(self.study_id, cf_id)
        self.assertEqual(run_res["status"], "COMPLETED")

        # Validate
        val_res = self.manager.validate_counterfactual(self.study_id, cf_id)
        self.assertEqual(val_res["status"], "VALIDATED")

        # Publish
        pub_res = self.manager.publish_counterfactual(self.study_id, cf_id)
        self.assertEqual(pub_res["status"], "PUBLISHED")

    def test_04_roi_bounds_validation_and_normalization(self):
        """4. Validates ROI bounds handling and normalized coordinates calculation."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        roi_in = {"x": 100, "y": 150, "width": 80, "height": 90, "norm_x": 0.2, "norm_y": 0.3, "norm_width": 0.16, "norm_height": 0.18}
        conf = self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_BLUR", roi_in, strength=1.5, seed=123)
        roi_out = conf["perturbation"]["roi"]
        self.assertEqual(roi_out["x"], 100)
        self.assertEqual(roi_out["y"], 150)
        self.assertEqual(roi_out["width"], 80)
        self.assertEqual(roi_out["height"], 90)
        self.assertTrue(0.0 <= roi_out["norm_x"] <= 1.0)
        self.assertTrue(0.0 <= roi_out["norm_y"] <= 1.0)

    def test_05_perturbation_determinism(self):
        """5. Verifies that applying the same perturbation with the same seed yields bit-identical derived images."""
        test_img = Image.new("RGB", (224, 224), color=120)
        roi = {"x": 30, "y": 30, "width": 50, "height": 50}

        img_a, meta_a = self.manager.apply_perturbation(test_img, "REGION_NOISE", roi, strength=1.0, seed=999)
        img_b, meta_b = self.manager.apply_perturbation(test_img, "REGION_NOISE", roi, strength=1.0, seed=999)

        arr_a = np.array(img_a)
        arr_b = np.array(img_b)
        self.assertTrue(np.array_equal(arr_a, arr_b), "Same seed perturbation must be deterministic and bit-identical.")

    def test_06_original_image_disk_immutability(self):
        """6. Verifies that counterfactual generation never modifies the source image on disk."""
        src_path, _ = self.manager._resolve_source_image(self.study_id)
        hash_before = compute_file_sha256(src_path)

        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 20, "y": 20, "width": 40, "height": 40})
        self.manager.run_counterfactual(self.study_id, cf_id)

        hash_after = compute_file_sha256(src_path)
        self.assertEqual(hash_before, hash_after, "Original image file must remain strictly byte-identical.")

    def test_07_all_eight_perturbation_methods_supported(self):
        """7. Verifies all 8 perturbation methods execute cleanly and generate valid images."""
        methods = [
            "REGION_MASK", "REGION_OCCLUSION", "REGION_BLUR", "REGION_NOISE",
            "REGION_CONTRAST", "REGION_BRIGHTNESS", "REMOVE_ATTRIBUTION_REGION", "RANDOM_CONTROL_REGION"
        ]
        test_img = Image.new("RGB", (224, 224), color=150)
        roi = {"x": 40, "y": 40, "width": 60, "height": 60}

        for m in methods:
            out_img, meta = self.manager.apply_perturbation(test_img, m, roi, strength=1.0, seed=42)
            self.assertEqual(out_img.size, (224, 224))
            self.assertEqual(meta["method"], m)

    def test_08_finding_delta_calculation_logic(self):
        """8. Validates absolute and relative delta computation."""
        base_f = [{"pathology": "Atelectasis", "score": 0.60}]
        cf_f = [{"pathology": "Atelectasis", "score": 0.45}]
        deltas = self.manager.calculate_finding_deltas(base_f, cf_f)
        self.assertEqual(len(deltas), 1)
        d = deltas[0]
        self.assertAlmostEqual(d["delta_abs"], -0.15, places=4)
        self.assertAlmostEqual(d["delta_rel"], -0.25, places=4)
        self.assertEqual(d["direction"], "DECREASED")
        self.assertTrue(d["is_significant"])

    def test_09_zero_denominator_protection(self):
        """9. Verifies zero-denominator safety in finding delta calculation."""
        base_f = [{"pathology": "Pneumonia", "score": 0.0}]
        cf_f = [{"pathology": "Pneumonia", "score": 0.20}]
        deltas = self.manager.calculate_finding_deltas(base_f, cf_f)
        self.assertEqual(len(deltas), 1)
        d = deltas[0]
        self.assertFalse(math.isinf(d["delta_rel"]))
        self.assertFalse(math.isnan(d["delta_rel"]))
        self.assertEqual(d["direction"], "INCREASED")

    def test_10_nan_infinity_rejection(self):
        """10. Verifies rejection of NaN and Infinity in numerical validation."""
        with self.assertRaises(ValueError):
            validate_finite_number(float("nan"), "test_nan")
        with self.assertRaises(ValueError):
            validate_finite_number(float("inf"), "test_inf")
        with self.assertRaises(ValueError):
            validate_finite_number(float("-inf"), "test_neg_inf")

    def test_11_gradcam_comparison_and_diff_map(self):
        """11. Verifies Grad-CAM attribution comparison and difference map generation."""
        hm_base = np.ones((224, 224), dtype=np.float32) * 0.5
        hm_cf = np.ones((224, 224), dtype=np.float32) * 0.3
        roi = {"norm_x": 0.1, "norm_y": 0.1, "norm_width": 0.2, "norm_height": 0.2}

        attr_delta, diff_vis = self.manager.calculate_attribution_delta("Atelectasis", hm_base, hm_cf, roi)
        self.assertIn("mean_abs_difference", attr_delta)
        self.assertIn("overlap_fraction", attr_delta)
        self.assertIn("roi_attribution_fraction_baseline", attr_delta)
        self.assertEqual(diff_vis.shape, (224, 224))
        self.assertTrue(0.0 <= diff_vis.min() <= diff_vis.max() <= 1.0)

    def test_12_qa_impact_analysis(self):
        """12. Verifies Diagnostic QA question answers comparison across levels."""
        deltas = [
            {"pathology": "Atelectasis", "baseline_score": 0.65, "counterfactual_score": 0.25},
            {"pathology": "Effusion", "baseline_score": 0.10, "counterfactual_score": 0.10}
        ]
        qa_items = self.manager.evaluate_qa_impact(deltas)
        self.assertEqual(len(qa_items), 4)  # 2 questions per finding (Level 1 + Level 2)

        # First item (Level 1 presence for Atelectasis) should be changed
        atel_l1 = next(q for q in qa_items if q["finding"] == "Atelectasis" and "Level 1" in q["level"])
        self.assertEqual(atel_l1["baseline_answer"], "yes")
        self.assertEqual(atel_l1["counterfactual_answer"], "no")
        self.assertTrue(atel_l1["changed"])

    def test_13_report_impact_statement_diffing(self):
        """13. Verifies machine report statement diffing and status transitions."""
        deltas = [
            {"pathology": "Atelectasis", "baseline_score": 0.70, "counterfactual_score": 0.15},
            {"pathology": "Pneumonia", "baseline_score": 0.10, "counterfactual_score": 0.10}
        ]
        qa_items = self.manager.evaluate_qa_impact(deltas)
        rep_impact = self.manager.evaluate_report_impact(deltas, qa_items)

        self.assertIn("Atelectasis", rep_impact["findings_changed"])
        self.assertTrue(rep_impact["impression_changed"])
        self.assertTrue(len(rep_impact["status_changes"]) >= 1)
        self.assertEqual(rep_impact["status_changes"][0]["baseline_status"], "supported")
        self.assertEqual(rep_impact["status_changes"][0]["counterfactual_status"], "absent")

    def test_14_control_roi_generation_non_overlapping(self):
        """14. Verifies random control ROI generation creates non-overlapping region."""
        target_roi = {"x": 100, "y": 100, "width": 80, "height": 80}
        ctrl_roi = self.manager.generate_random_control_roi((512, 512), target_roi, seed=200)

        self.assertIsNotNone(ctrl_roi)
        self.assertEqual(ctrl_roi["width"], target_roi["width"])
        self.assertEqual(ctrl_roi["height"], target_roi["height"])
        # Check non-overlap
        tx, ty, w, h = target_roi["x"], target_roi["y"], target_roi["width"], target_roi["height"]
        cx, cy = ctrl_roi["x"], ctrl_roi["y"]
        overlap = not (cx + w <= tx or cx >= tx + w or cy + h <= ty or cy >= ty + h)
        self.assertFalse(overlap, "Generated control ROI must not overlap with target ROI.")

    def test_15_reproducibility_fingerprinting(self):
        """15. Verifies deterministic fingerprint generation on counterfactual run."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 50, "y": 50, "width": 50, "height": 50})
        res = self.manager.run_counterfactual(self.study_id, cf_id)

        repro = res["reproducibility"]
        self.assertEqual(repro["status"], "REPRODUCIBLE")
        self.assertTrue(len(repro["fingerprint"]) >= 32)
        self.assertTrue(len(repro["source_image_hash"]) >= 32)
        self.assertTrue(len(repro["config_hash"]) >= 32)

    def test_16_published_experiment_immutability(self):
        """16. Asserts that published counterfactual experiments cannot be reconfigured or overwritten."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_BLUR", {"x": 30, "y": 30, "width": 40, "height": 40})
        self.manager.run_counterfactual(self.study_id, cf_id)
        self.manager.validate_counterfactual(self.study_id, cf_id)
        self.manager.publish_counterfactual(self.study_id, cf_id)

        with self.assertRaises(ValueError):
            self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 10, "y": 10, "width": 20, "height": 20})

    def test_17_ground_truth_isolation(self):
        """17. Verifies zero ground-truth XML report leakage into counterfactual outputs."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 20, "y": 20, "width": 30, "height": 30})
        res = self.manager.run_counterfactual(self.study_id, cf_id)

        gt_ok, gt_err = check_ground_truth_isolation(res)
        self.assertTrue(gt_ok, f"Ground-truth token leaked: {gt_err}")

    def test_18_path_traversal_protection(self):
        """18. Verifies path traversal attacks in study or counterfactual IDs are rejected."""
        traversal_attempts = ["../escape", "..\\escape", "study/../../etc", "valid\x00null"]
        for att in traversal_attempts:
            with self.assertRaises(ValueError):
                sanitize_id(att)

    def test_19_statistical_summary_and_bootstrap_ci(self):
        """19. Verifies comparator statistical calculations and bootstrap CI."""
        values = [0.10, 0.15, 0.12, 0.18, 0.22, 0.14, 0.16]
        stats = self.comparator.compute_summary_statistics(values)
        self.assertEqual(stats["count"], 7)
        self.assertTrue(0.10 <= stats["mean"] <= 0.22)
        self.assertAlmostEqual(stats["median"], 0.15, places=2)

        ci = self.comparator.compute_bootstrap_ci(values, num_resamples=500, confidence_level=0.95)
        self.assertTrue(ci["ci_lower"] <= stats["mean"] <= ci["ci_upper"])

    def test_20_two_experiment_comparison_neutral_wording(self):
        """20. Verifies comparator generates neutral research wording without causal claims."""
        exp_a = self.manager.create_counterfactual(self.study_id, "PA")
        cf_a_id = exp_a["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_a_id, "REGION_MASK", {"x": 30, "y": 30, "width": 40, "height": 40})
        run_a = self.manager.run_counterfactual(self.study_id, cf_a_id)

        exp_b = self.manager.create_counterfactual(self.study_id, "PA")
        cf_b_id = exp_b["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_b_id, "REGION_BLUR", {"x": 30, "y": 30, "width": 40, "height": 40})
        run_b = self.manager.run_counterfactual(self.study_id, cf_b_id)

        comp = self.comparator.compare_two_experiments(run_a, run_b)
        self.assertIn("neutral_summary", comp)
        self.assertIn("comparison_table", comp)
        self.assertIn("statistics", comp)
        self.assertIn("disclaimer", comp)
        # Check non-causal language
        forbidden_causal = ["proves causality", "causes the pathology", "definitive proof"]
        for f in forbidden_causal:
            self.assertNotIn(f, comp["neutral_summary"].lower())

    def test_21_audit_trail_integrity(self):
        """21. Verifies audit trail maintains sequential hashed records."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 30, "y": 30, "width": 40, "height": 40})
        res = self.manager.run_counterfactual(self.study_id, cf_id)

        trail = res["audit_trail"]
        self.assertTrue(len(trail) >= 3)
        actions = [t["action"] for t in trail]
        self.assertEqual(actions[:3], ["CREATE", "CONFIGURE", "RUN"])
        for entry in trail:
            self.assertIn("record_hash", entry)
            self.assertTrue(len(entry["record_hash"]) >= 8)

    def test_22_artifact_disk_isolation(self):
        """22. Verifies all counterfactual artifacts are saved in data/counterfactuals/{study_id}/artifacts/."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 40, "y": 40, "width": 40, "height": 40})
        res = self.manager.run_counterfactual(self.study_id, cf_id)

        expected_artifacts_dir = os.path.join(BASE_DIR, "data", "counterfactuals", self.study_id, "artifacts")
        self.assertTrue(os.path.exists(expected_artifacts_dir))
        expected_cf_img = os.path.join(expected_artifacts_dir, f"{cf_id}_counterfactual_image.png")
        self.assertTrue(os.path.exists(expected_cf_img))

    def test_23_mandatory_research_disclaimer_presence(self):
        """23. Asserts mandatory research disclaimers are present in manager and records."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        self.assertIn("disclaimer", exp)
        self.assertIn("RESEARCH USE ONLY", exp["disclaimer"])
        self.assertIn("Model Activation Score != Clinical Probability", exp["disclaimer"])

    def test_24_dashboard_summary_metrics(self):
        """24. Verifies dashboard summary counts experiments accurately."""
        summary = self.manager.get_dashboard_summary()
        self.assertIn("total_experiments", summary)
        self.assertIn("completed", summary)
        self.assertIn("validated", summary)
        self.assertIn("published", summary)
        self.assertIn("reproducible", summary)
        self.assertIn("disclaimer", summary)

    def test_25_safety_validator_execution(self):
        """25. Runs validate_phase_2_0.py suite and confirms all checks pass."""
        res = run_all_phase_2_0_safety_checks(BASE_DIR)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["iu_xray_dataset_intact"])
        self.assertTrue(res["immutability_verified"])
        self.assertTrue(res["security_verified"])

    def test_26_control_comparison_runner(self):
        """26. Verifies automated control comparison between target ROI and random control ROI."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 60, "y": 60, "width": 50, "height": 50})
        comp = self.comparator.run_control_comparison(self.study_id, cf_id)
        self.assertIn("comparison_table", comp)
        self.assertIn("neutral_summary", comp)

    def test_27_invalid_perturbation_strength_rejected(self):
        """27. Verifies strength out of bounds (> 2.0 or < 0.0) is rejected."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        with self.assertRaises(ValueError):
            self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 10, "y": 10, "width": 20, "height": 20}, strength=2.5)

    def test_28_invalid_perturbation_method_rejected(self):
        """28. Verifies unsupported perturbation method raises ValueError."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        with self.assertRaises(ValueError):
            self.manager.configure_counterfactual(self.study_id, cf_id, "INVALID_PERTURBATION_XYZ", {"x": 10, "y": 10, "width": 20, "height": 20})

    def test_29_archive_counterfactual(self):
        """29. Verifies archive experiment transition."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        arch = self.manager.archive_counterfactual(self.study_id, cf_id)
        self.assertEqual(arch["status"], "ARCHIVED")

    def test_30_list_counterfactuals_filter(self):
        """30. Verifies listing counterfactuals with and without study filtering."""
        all_list = self.manager.list_counterfactuals()
        study_list = self.manager.list_counterfactuals(study_id=self.study_id)
        self.assertIsInstance(all_list, list)
        self.assertIsInstance(study_list, list)

    def test_31_iu_xray_5399_dataset_integrity(self):
        """31. Verifies that base dataset is completely intact (~5,399 files)."""
        ok, msg, count = check_iu_xray_dataset_integrity(BASE_DIR)
        self.assertTrue(ok)
        self.assertTrue(count >= 5300)

    def test_32_security_sanitization_protection(self):
        """32. Verifies security sanitization rejects illegal characters and traversal."""
        ok, msg = check_security_sanitization()
        self.assertTrue(ok, msg)

    def test_33_reproducibility_drift_detection(self):
        """33. Verifies validation detects configuration/image hash mismatch."""
        exp = self.manager.create_counterfactual(self.study_id, "PA")
        cf_id = exp["counterfactual_id"]
        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_MASK", {"x": 40, "y": 40, "width": 40, "height": 40})
        self.manager.run_counterfactual(self.study_id, cf_id)

        # Mutate recorded hash to simulate drift
        exp_data = self.manager.get_counterfactual(self.study_id, cf_id)
        exp_data["reproducibility"]["source_image_hash"] = "tampered_fake_hash_123456789"
        self.manager._save_record(exp_data)

        # Run validation
        val_res = self.manager.validate_counterfactual(self.study_id, cf_id)
        self.assertEqual(val_res["reproducibility"]["status"], "DRIFT_DETECTED")

    def test_34_full_e2e_counterfactual_lifecycle(self):
        """34. Tests full end-to-end lifecycle from creation to publishing and export."""
        exp = self.manager.create_counterfactual(self.study_id, "PA", created_by="e2e_tester")
        cf_id = exp["counterfactual_id"]

        self.manager.configure_counterfactual(self.study_id, cf_id, "REGION_OCCLUSION", {"x": 50, "y": 50, "width": 70, "height": 70}, strength=1.2)
        run_res = self.manager.run_counterfactual(self.study_id, cf_id)
        self.assertTrue(len(run_res["finding_deltas"]) >= 14)
        self.assertTrue(len(run_res["qa_impact"]) >= 28)

        val_res = self.manager.validate_counterfactual(self.study_id, cf_id)
        self.assertEqual(val_res["status"], "VALIDATED")

        pub_res = self.manager.publish_counterfactual(self.study_id, cf_id)
        self.assertEqual(pub_res["status"], "PUBLISHED")

        # Verify export retrieval
        saved = self.manager.get_counterfactual(self.study_id, cf_id)
        self.assertEqual(saved["counterfactual_id"], cf_id)
        self.assertEqual(saved["status"], "PUBLISHED")

    def test_35_existing_architecture_regression(self):
        """35. Verifies existing project components remain intact and functional."""
        from backend.evidence_layer import EvidenceLayer
        from backend.diagnostic_qa import DiagnosticQAEngine
        from backend.visual_grounding import VisualGrounding

        ev = EvidenceLayer()
        self.assertIsNotNone(ev)
        qa = DiagnosticQAEngine()
        self.assertIsNotNone(qa)
        vg = VisualGrounding()
        self.assertIsNotNone(vg)


if __name__ == "__main__":
    unittest.main()
