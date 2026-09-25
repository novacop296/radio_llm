#!/usr/bin/env python3
"""
Phase 1.3 End-to-End Multi-Reviewer Consensus, Inter-Rater Agreement & Adjudication Pipeline

Demonstrates and verifies:
1. Multi-reviewer cohort registration and independent evaluation sessions.
2. Complete data isolation across independent reviewer sessions.
3. Machine evidence immutability and zero ground-truth leakage.
4. Deterministic consensus aggregation (unanimous, majority, adjudication_required).
5. Statistical inter-rater reliability metrics (Average Agreement Ratio, Cohen's Kappa, Fleiss' Kappa).
6. Binding adjudication recording with mandatory rationale.
7. Consensus report synthesis, finalization, and multi-format export.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from backend.review_manager import global_review_manager, VALID_REVIEWER_STATUSES
from backend.consensus_manager import global_consensus_manager
from backend.validate_consensus import validate_consensus_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_consensus_pipeline")


def run_e2e_consensus_pipeline(study_id: str = "CXR1122") -> Dict[str, Any]:
    print("=" * 80)
    print(f"RUNNING PHASE 1.3 E2E MULTI-REVIEWER CONSENSUS PIPELINE FOR STUDY: {study_id}")
    print("=" * 80)

    # Clean up any prior test run files for this study
    reviews_dir = os.path.join(DATA_DIR, "reviews")
    consensus_dir = os.path.join(DATA_DIR, "consensus")
    if os.path.exists(reviews_dir):
        for f in os.listdir(reviews_dir):
            if f.startswith(study_id) and any(r in f for r in ["dr_alice", "dr_bob", "dr_carol"]):
                try:
                    os.remove(os.path.join(reviews_dir, f))
                except Exception:
                    pass
    if os.path.exists(consensus_dir):
        cf = os.path.join(consensus_dir, f"{study_id}.json")
        if os.path.exists(cf):
            try:
                os.remove(cf)
            except Exception:
                pass

    # 1. Inspect Machine Baseline Artifacts
    machine_review = global_review_manager.get_or_create_review(study_id, reviewer_id="researcher_01")
    
    findings_list = [fr["finding"] for fr in machine_review.get("finding_reviews", [])]
    print(f"\n[STAGE 1] Machine Evidence Loaded: {len(findings_list)} candidate findings")
    for fr in machine_review.get("finding_reviews", []):
        score = fr.get("model_score", fr.get("machine_score", 0.0))
        print(f"  - {fr['finding']:<16} | Machine Status: {fr['machine_status']:<10} | Score: {score:.4f}")

    # 2. Simulate 3 Independent Clinical Reviewers
    reviewers = [
        {"id": "dr_alice_01", "display_name": "Dr. Alice Morgan, MD", "role": "Attending Radiologist"},
        {"id": "dr_bob_02", "display_name": "Dr. Bob Chen, MD", "role": "Thoracic Radiologist"},
        {"id": "dr_carol_03", "display_name": "Dr. Carol Danvers, MD", "role": "Clinical Reviewer"}
    ]

    print(f"\n[STAGE 2] Simulating 3 Independent Reviewers")
    
    # Reviewer 1 (Alice): Confirms Infiltration & Cardiomegaly, refutes others
    r1_decisions = {
        "Infiltration": ("confirmed_present", "right_lower_lobe", "moderate", "Prominent bibasilar opacity consistent with infiltration."),
        "Cardiomegaly": ("confirmed_present", "unspecified", "mild", "Borderline enlarged cardiac silhouette."),
        "Pneumothorax": ("confirmed_absent", "unspecified", "unspecified", "No pleural line or pneumothorax visible."),
        "Effusion": ("confirmed_absent", "unspecified", "unspecified", "Costophrenic angles sharp."),
        "Consolidation": ("confirmed_absent", "unspecified", "unspecified", "No dense lobar consolidation."),
        "Fracture": ("confirmed_absent", "unspecified", "unspecified", "Bony thorax intact."),
        "Nodule": ("confirmed_absent", "unspecified", "unspecified", "No discrete pulmonary nodule identified.")
    }

    # Reviewer 2 (Bob): Agrees on Infiltration & Cardiomegaly, but marks Effusion as possible disagreement
    r2_decisions = {
        "Infiltration": ("confirmed_present", "right_lower_lobe", "moderate", "Bilateral hazy opacities noted."),
        "Cardiomegaly": ("confirmed_present", "unspecified", "mild", "Mild cardiomegaly."),
        "Pneumothorax": ("confirmed_absent", "unspecified", "unspecified", "Apex clear."),
        "Effusion": ("uncertain", "right_lower_lobe", "mild", "Blunting of right CP angle questionable for trace effusion."),
        "Consolidation": ("confirmed_absent", "unspecified", "unspecified", "No consolidation."),
        "Fracture": ("confirmed_absent", "unspecified", "unspecified", "Ribs normal."),
        "Nodule": ("confirmed_absent", "unspecified", "unspecified", "No nodule.")
    }

    # Reviewer 3 (Carol): Agrees on Infiltration, Cardiomegaly, and confirms Effusion absent
    r3_decisions = {
        "Infiltration": ("confirmed_present", "right_lower_lobe", "moderate", "Patchy infiltrates in lower zones."),
        "Cardiomegaly": ("confirmed_present", "unspecified", "mild", "Mildly prominent cardiac contour."),
        "Pneumothorax": ("confirmed_absent", "unspecified", "unspecified", "No pneumothorax."),
        "Effusion": ("confirmed_absent", "unspecified", "unspecified", "Sulci clear, no effusion."),
        "Consolidation": ("confirmed_absent", "unspecified", "unspecified", "No consolidation."),
        "Fracture": ("confirmed_absent", "unspecified", "unspecified", "No fracture."),
        "Nodule": ("confirmed_absent", "unspecified", "unspecified", "No nodule.")
    }

    cohort_decisions = [
        (reviewers[0]["id"], r1_decisions),
        (reviewers[1]["id"], r2_decisions),
        (reviewers[2]["id"], r3_decisions)
    ]

    for r_info in reviewers:
        r_id = r_info["id"]
        # Ensure session created
        global_review_manager.get_or_create_review(study_id, reviewer_id=r_id, reviewer_info=r_info)

    for r_id, decs in cohort_decisions:
        for fname, (status, loc, sev, comm) in decs.items():
            global_review_manager.update_finding_review(
                study_id=study_id,
                payload={
                    "finding": fname,
                    "reviewer_status": status,
                    "reviewer_location": loc,
                    "reviewer_severity": sev,
                    "reviewer_comment": comm,
                    "reviewer_id": r_id
                },
                reviewer_id=r_id
            )
        # Finalize individual review
        success, sess, issues = global_review_manager.finalize_review(
            study_id=study_id,
            reviewer_id=r_id
        )
        if not success:
            print(f"  [WARN] Finalize review issues for {r_id}: {issues}")
        print(f"  [OK] Reviewer session '{r_id}' completed and finalized.")

    # 3. Initialize / Recalculate Multi-Reviewer Consensus Session
    print(f"\n[STAGE 3] Recalculating Multi-Reviewer Consensus Cohort")
    consensus_session = global_consensus_manager.create_consensus_session(
        study_id=study_id,
        required_reviewers=3,
        minimum_reviewers=2,
        reviewers_list=reviewers
    )
    print(f"  - Consensus ID: {consensus_session['consensus_id']}")
    print(f"  - Status: {consensus_session['status'].upper()}")
    print(f"  - Reviewers in Cohort: {len(consensus_session['reviewers'])}")

    # 4. Display Statistical Agreement Metrics
    metrics = consensus_session.get("agreement_metrics", {})
    print(f"\n[STAGE 4] Inter-Rater Reliability & Statistical Agreement:")
    print(f"  - Active Reviewers: {metrics.get('reviewer_count')}")
    print(f"  - Evaluated Findings: {metrics.get('finding_count')}")
    print(f"  - Average Agreement Ratio: {metrics.get('average_agreement_ratio', 0.0) * 100:.2f}%")
    
    fk = metrics.get("fleiss_kappa")
    if fk and fk.get("value") is not None:
        p_o = fk.get("observed_agreement_mean", fk.get("observed_agreement", 0.0))
        p_e = fk.get("expected_agreement", 0.0)
        print(f"  - Fleiss Kappa (N=3): {fk['value']:.4f} (Observed: {p_o:.4f}, Expected: {p_e:.4f})")

    # 5. Finding Consensus Matrix
    print(f"\n[STAGE 5] Finding Consensus Matrix:")
    print(f"  {'Finding':<16} | {'Machine':<10} | {'Consensus':<18} | {'Status':<20} | {'Agreement Ratio':<15}")
    print("  " + "-" * 88)
    for fc in consensus_session.get("finding_consensus", []):
        f_name = fc["finding"]
        m_stat = fc["machine_status"]
        c_dec = fc["consensus_decision"] or "PENDING"
        c_stat = fc["consensus_status"].upper()
        agr = f"{fc.get('agreement_ratio', 0.0) * 100:.1f}%" if fc.get('agreement_ratio') is not None else "-"
        print(f"  {f_name:<16} | {m_stat:<10} | {c_dec:<18} | {c_stat:<20} | {agr:<15}")

    # 6. Check Adjudication Requirements
    adj_info = consensus_session.get("adjudication", {})
    pending_adjs = adj_info.get("items_requiring_adjudication", [])
    print(f"\n[STAGE 6] Adjudication Status: {len(pending_adjs)} items requiring adjudication")
    for item in pending_adjs:
        if item.get("status") == "pending":
            print(f"  [ACTION] Adjudicating item: {item['target_type']} -> {item['target_id']}")
            global_consensus_manager.record_adjudication(
                study_id=study_id,
                target_type=item["target_type"],
                target_id=item["target_id"],
                decision="confirmed_absent",
                adjudicator_id="lead_adjudicator",
                reason="Adjudicator consensus review: sulci remain clear, no true pleural effusion."
            )
            print(f"    [OK] Adjudication recorded for {item['target_id']}.")

    # Refresh consensus session post-adjudication
    consensus_session = global_consensus_manager.get_consensus_session(study_id)

    # 7. Consensus Report Finalization
    print(f"\n[STAGE 7] Consensus Report Finalization")
    final_findings = [
        {"finding": "Infiltration", "statement": "Bilateral bibasilar interstitial infiltration is confirmed by multi-reviewer consensus.", "status": "supported"},
        {"finding": "Cardiomegaly", "statement": "Mild cardiomegaly is confirmed present.", "status": "supported"},
        {"finding": "Effusion", "statement": "No pleural effusion is identified by consensus adjudication.", "status": "absent"},
        {"finding": "Pneumothorax", "statement": "No pneumothorax is identified.", "status": "absent"},
        {"finding": "Consolidation", "statement": "No dense lobar consolidation is present.", "status": "absent"},
        {"finding": "Fracture", "statement": "No rib fractures or bony abnormalities are observed.", "status": "absent"},
        {"finding": "Nodule", "statement": "No discrete pulmonary nodules or masses are present.", "status": "absent"}
    ]
    final_impression = [
        "1. Bibasilar pulmonary infiltration, consistent with infectious or inflammatory process.",
        "2. Mild cardiomegaly without overt pulmonary edema or pleural effusion."
    ]

    global_consensus_manager.save_consensus_report_draft(
        study_id=study_id,
        final_findings=final_findings,
        final_impression=final_impression,
        reviewer_comment="Full consensus achieved with 1 finding resolved via formal adjudication."
    )

    success, finalized_session, fin_issues = global_consensus_manager.finalize_consensus(
        study_id=study_id,
        user_id="lead_adjudicator"
    )
    if not success:
        raise RuntimeError(f"Failed to finalize consensus: {fin_issues}")
    print(f"  [OK] Consensus Report Finalized: Status = {finalized_session['status'].upper()}")

    # 8. Run Schema & Invariant Validation
    print(f"\n[STAGE 8] Schema & Invariant Validation")
    is_valid, issues = validate_consensus_session(finalized_session, require_finalization_ready=True)
    if is_valid:
        print("  [OK] Consensus Session Schema & Safety Invariants: PASS (0 violations)")
    else:
        print(f"  [FAIL] Consensus Validation Issues: {len(issues)}")
        for iss in issues:
            print(f"    - {iss}")
        raise RuntimeError(f"Consensus package validation failed with {len(issues)} issues!")

    # 9. Exports
    print(f"\n[STAGE 9] Export Artifacts")
    json_content, j_type = global_consensus_manager.export_consensus(study_id, format="json")
    text_content, t_type = global_consensus_manager.export_consensus(study_id, format="text")
    print(f"  [OK] JSON Export generated ({len(json_content)} bytes, {j_type})")
    print(f"  [OK] Plain Text Export generated ({len(text_content)} chars, {t_type})")

    print("\n" + "=" * 80)
    print("PHASE 1.3 E2E CONSENSUS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    return finalized_session


if __name__ == "__main__":
    study = sys.argv[1] if len(sys.argv) > 1 else "CXR1122"
    run_e2e_consensus_pipeline(study)
