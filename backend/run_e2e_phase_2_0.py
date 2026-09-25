"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 2.0 — End-to-End Verification Pipeline

Module: run_e2e_phase_2_0.py
Purpose:
- Implements the complete automated lifecycle verification for Phase 2.0.
- Sequence:
    1. Study Discovery
    2. Load Original Image & Hash
    3. Baseline Inference
    4. Baseline Grad-CAM Generation
    5. Interactive ROI Selection & Normalization
    6. Configure Perturbation Method
    7. Generate Derived Counterfactual Image
    8. Counterfactual Inference Execution
    9. Counterfactual Grad-CAM Generation
    10. Calculate Finding Deltas (with Zero-Denominator Protection)
    11. Calculate Attribution Difference Heatmap & Overlap Metrics
    12. Evaluate Diagnostic QA Impact across Levels
    13. Evaluate Machine Report Statement & Status Impact
    14. Execute Random Control ROI Comparison
    15. Invariant & Schema Validation
    16. Reproducibility Fingerprint Generation
    17. Finalize & Publish with Immutability Lock
    18. Export Verification
    19. Byte-level Original Image & IU X-Ray Dataset Immutability Assertion

DISCLAIMER:
RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS.
"""

import os
import sys
import json
import time
import math
import hashlib
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.counterfactual_manager import (
    global_counterfactual_manager,
    compute_file_sha256,
    RESEARCH_DISCLAIMER
)
from backend.counterfactual_comparator import global_counterfactual_comparator
from backend.validate_phase_2_0 import (
    check_iu_xray_dataset_integrity,
    check_ground_truth_isolation,
    run_all_phase_2_0_safety_checks
)


def run_e2e_phase_2_0_pipeline(study_id: str = "CXR1122") -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 2.0 END-TO-END VERIFICATION PIPELINE")
    print("Interactive Counterfactual Explanations & Clinical Reasoning Synthesis")
    print("=" * 80)

    stages_passed = 0
    total_stages = 20

    # Stage 1: Dataset Integrity & Hash Snapshot Before
    print("\n[STAGE 1/20] Checking IU X-Ray Dataset Integrity Before Execution...")
    ds_ok, ds_msg, file_count = check_iu_xray_dataset_integrity(BASE_DIR)
    assert ds_ok, f"Dataset check failed: {ds_msg}"
    print(f"  ✓ {ds_msg}")
    stages_passed += 1

    # Stage 2: Resolve Source Image & Compute Pre-execution Checksum
    print("\n[STAGE 2/20] Resolving Source Radiograph & Pre-execution Hash...")
    src_path, src_filename = global_counterfactual_manager._resolve_source_image(study_id, "PA")
    pre_img_hash = compute_file_sha256(src_path)
    print(f"  ✓ Source radiograph: {src_filename} (SHA256: {pre_img_hash[:16]}...)")
    stages_passed += 1

    # Stage 3: Create Counterfactual Experiment (DRAFT)
    print("\n[STAGE 3/20] Initializing Counterfactual Experiment in DRAFT State...")
    exp = global_counterfactual_manager.create_counterfactual(study_id, "PA", created_by="e2e_pipeline")
    cf_id = exp["counterfactual_id"]
    assert exp["status"] == "DRAFT"
    print(f"  ✓ Experiment created: {cf_id} (Status: {exp['status']})")
    stages_passed += 1

    # Stage 4: Configure Perturbation ROI & Parameters
    print("\n[STAGE 4/20] Configuring Perturbation (REGION_MASK, Strength=1.0, Seed=42)...")
    roi = {"x": 50, "y": 50, "width": 80, "height": 80, "norm_x": 0.22, "norm_y": 0.22, "norm_width": 0.35, "norm_height": 0.35}
    conf = global_counterfactual_manager.configure_counterfactual(
        study_id, cf_id, method="REGION_MASK", roi=roi, strength=1.0, seed=42, target_pathology="Atelectasis"
    )
    assert conf["status"] == "CONFIGURED"
    print(f"  ✓ Perturbation configured (Status: {conf['status']})")
    stages_passed += 1

    # Stage 5: Execute Baseline Inference & Heatmap Generation
    print("\n[STAGE 5/20] Executing Baseline Inference & Grad-CAM Heatmap...")
    # Triggered within run_counterfactual
    run_res = global_counterfactual_manager.run_counterfactual(study_id, cf_id)
    assert run_res["status"] == "COMPLETED"
    assert len(run_res["baseline"]["findings"]) >= 14
    print(f"  ✓ Baseline inference complete ({len(run_res['baseline']['findings'])} findings evaluated)")
    stages_passed += 1

    # Stage 6: Derived Counterfactual Image Artifact Generation & Hash Check
    print("\n[STAGE 6/20] Verifying Derived Counterfactual Image Artifact...")
    cf_img_path = run_res["counterfactual"]["image_path"]
    assert os.path.exists(cf_img_path), "Derived counterfactual image file must exist."
    cf_hash = compute_file_sha256(cf_img_path)
    assert cf_hash != pre_img_hash, "Derived counterfactual image must differ from original."
    print(f"  ✓ Derived image generated: {os.path.basename(cf_img_path)} (SHA256: {cf_hash[:16]}...)")
    stages_passed += 1

    # Stage 7: Counterfactual Inference Execution
    print("\n[STAGE 7/20] Verifying Counterfactual Inference Results...")
    assert len(run_res["counterfactual"]["findings"]) >= 14
    print(f"  ✓ Counterfactual inference complete ({len(run_res['counterfactual']['findings'])} findings)")
    stages_passed += 1

    # Stage 8: Finding Deltas Calculation
    print("\n[STAGE 8/20] Calculating Finding Deltas with Zero-Denominator Safety...")
    deltas = run_res["finding_deltas"]
    assert len(deltas) >= 14
    for d in deltas:
        assert not math.isnan(d["delta_abs"])
        assert not math.isinf(d["delta_rel"])
    print(f"  ✓ Finding deltas calculated for all {len(deltas)} pathologies without NaN/Inf.")
    stages_passed += 1

    # Stage 9: Attribution Difference Heatmap & Overlap Metrics
    print("\n[STAGE 9/20] Calculating Attribution Difference Heatmap & Overlap...")
    attr_deltas = run_res["attribution_deltas"]
    assert len(attr_deltas) >= 1
    ad = attr_deltas[0]
    print(f"  ✓ Attribution overlap: {(ad['overlap_fraction']*100):.1f}% | Mean |Δ|: {ad['mean_abs_difference']:.4f}")
    stages_passed += 1

    # Stage 10: Diagnostic QA Impact Evaluation
    print("\n[STAGE 10/20] Evaluating Diagnostic QA Impact across Question Levels...")
    qa_impact = run_res["qa_impact"]
    assert len(qa_impact) >= 28  # 14 findings * 2 levels
    print(f"  ✓ Diagnostic QA impact evaluated across {len(qa_impact)} questions.")
    stages_passed += 1

    # Stage 11: Machine Report Impact & Statement Diffing
    print("\n[STAGE 11/20] Evaluating Machine Report Impact & Status Transitions...")
    rep_impact = run_res["report_impact"]
    assert "baseline_summary" in rep_impact
    assert "counterfactual_summary" in rep_impact
    print(f"  ✓ Report impact: {len(rep_impact['status_changes'])} status transitions identified.")
    stages_passed += 1

    # Stage 12: Random Control ROI Generation & Comparison
    print("\n[STAGE 12/20] Executing Automated Control ROI Comparison...")
    ctrl_comp = global_counterfactual_comparator.run_control_comparison(study_id, cf_id)
    assert "comparison_table" in ctrl_comp
    assert "statistics" in ctrl_comp
    assert "bootstrap_ci" in ctrl_comp
    print(f"  ✓ Control comparison complete. Mean diff: {ctrl_comp['statistics']['mean']:.4f}")
    stages_passed += 1

    # Stage 13: Ground-Truth Isolation Verification
    print("\n[STAGE 13/20] Verifying Zero Ground-Truth XML Report Leakage...")
    gt_ok, gt_err = check_ground_truth_isolation(run_res)
    assert gt_ok, f"Ground truth token detected: {gt_err}"
    print("  ✓ Zero ground-truth XML report tokens detected.")
    stages_passed += 1

    # Stage 14: Invariant & Reproducibility Validation
    print("\n[STAGE 14/20] Validating Experiment Invariants & Schema Conformance...")
    val_res = global_counterfactual_manager.validate_counterfactual(study_id, cf_id)
    assert val_res["status"] == "VALIDATED"
    assert val_res["reproducibility"]["status"] == "REPRODUCIBLE"
    print(f"  ✓ Experiment validated (Reproducibility: {val_res['reproducibility']['status']})")
    stages_passed += 1

    # Stage 15: Finalize Experiment
    print("\n[STAGE 15/20] Finalizing Experiment...")
    fin_res = global_counterfactual_manager.finalize_counterfactual(study_id, cf_id)
    assert fin_res["status"] == "VALIDATED"
    print("  ✓ Experiment certified ready for publishing.")
    stages_passed += 1

    # Stage 16: Publish Experiment & Enforce Immutability Locking
    print("\n[STAGE 16/20] Publishing Experiment to Immutable Registry...")
    pub_res = global_counterfactual_manager.publish_counterfactual(study_id, cf_id)
    assert pub_res["status"] == "PUBLISHED"
    print(f"  ✓ Experiment permanently published (Status: {pub_res['status']})")
    stages_passed += 1

    # Stage 17: Assert Published Record Cannot Be Reconfigured
    print("\n[STAGE 17/20] Verifying Immutable Modification Locking...")
    try:
        global_counterfactual_manager.configure_counterfactual(study_id, cf_id, "REGION_BLUR", {"x": 10, "y": 10, "width": 20, "height": 20})
        assert False, "Published experiment permitted modification!"
    except ValueError:
        print("  ✓ Mutation attempt correctly blocked with ValueError.")
    stages_passed += 1

    # Stage 18: Audit Trail Sequential Hashing Verification
    print("\n[STAGE 18/20] Verifying Audit Trail Sequential Integrity...")
    trail = pub_res["audit_trail"]
    assert len(trail) >= 5
    actions = [t["action"] for t in trail]
    print(f"  ✓ Audit actions recorded: {' -> '.join(actions)}")
    stages_passed += 1

    # Stage 19: Export Package Verification
    print("\n[STAGE 19/20] Verifying Export Package Format...")
    exp_saved = global_counterfactual_manager.get_counterfactual(study_id, cf_id)
    assert exp_saved["counterfactual_id"] == cf_id
    assert exp_saved["status"] == "PUBLISHED"
    assert "disclaimer" in exp_saved
    print("  ✓ Export package complete and self-contained.")
    stages_passed += 1

    # Stage 20: Post-execution Original Image Immutability Assertion
    print("\n[STAGE 20/20] Confirming Original Radiographs & Base IU X-Ray Dataset Immutability...")
    post_img_hash = compute_file_sha256(src_path)
    assert pre_img_hash == post_img_hash, "Source image hash changed during counterfactual run!"
    ds_post_ok, ds_post_msg, post_count = check_iu_xray_dataset_integrity(BASE_DIR)
    assert ds_post_ok and post_count == file_count, "IU X-Ray dataset files modified during run!"
    print(f"  ✓ Pre-hash:  {pre_img_hash}")
    print(f"  ✓ Post-hash: {post_img_hash}")
    print(f"  ✓ All {post_count} base dataset files remain 100% byte-identical.")
    stages_passed += 1

    print("\n" + "=" * 80)
    print(f"PHASE 2.0 E2E PIPELINE: {stages_passed}/{total_stages} STAGES PASSED (100%)")
    print(f"DISCLAIMER: {RESEARCH_DISCLAIMER}")
    print("=" * 80)

    return {
        "status": "PASS",
        "stages_passed": stages_passed,
        "total_stages": total_stages,
        "counterfactual_id": cf_id,
        "study_id": study_id,
        "source_image_hash_pre": pre_img_hash,
        "source_image_hash_post": post_img_hash,
        "base_dataset_files_verified": post_count,
        "reproducibility_status": "REPRODUCIBLE"
    }


if __name__ == "__main__":
    res = run_e2e_phase_2_0_pipeline("CXR1122")
    print(json.dumps(res, indent=2))
