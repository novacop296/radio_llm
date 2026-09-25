# Phase 1.3 — Multi-Reviewer Consensus, Inter-Rater Agreement & Adjudication

## 1. Executive Summary & Architecture Overview

Phase 1.3 extends the LLM-Assisted Explainable Radiology research prototype from single-clinician review (Phase 1.2) to **Multi-Reviewer Collaborative Evaluation, Inter-Rater Reliability Synthesis, and Dispute Adjudication**.

In high-stakes medical AI research, single-expert evaluations suffer from observer variability. Phase 1.3 provides a mathematically rigorous, auditable framework for aggregating evaluations from multiple independent clinical raters while preserving complete separation between:
1. **Authoritative Machine Evidence**: DenseNet-121 activations, Diagnostic QA answers, Evidence Layer states, and Grad-CAM spatial heatmaps.
2. **Individual Reviewer Sessions**: Isolated, independent evaluation workspaces stored per reviewer.
3. **Multi-Reviewer Consensus Synthesis**: Deterministic aggregation into unanimous or majority decisions with statistical agreement metrics.
4. **Adjudication Decisions**: Formal arbitration for diagnostic disputes with mandatory rationale and adjudicator provenance.
5. **Consensus Finalized Report**: Ratified clinical synthesis representing the research panel's unified findings.

```
+------------------------------------------------------------------------------------+
|                         AUTHORITATIVE MACHINE PIPELINE                             |
| DenseNet-121 (18 scores) -> Diagnostic QA Engine -> Evidence Layer -> Grad-CAM CAM |
+------------------------------------------------------------------------------------+
                                         |
                                (Read-Only Baseline)
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +------------------+           +------------------+
| Reviewer 1 (R1)  |           | Reviewer 2 (R2)  |           | Reviewer 3 (R3)  |
| data/reviews/    |           | data/reviews/    |           | data/reviews/    |
| CXR1122_rev01.json|          | CXR1122_rev02.json|          | CXR1122_rev03.json|
+------------------+           +------------------+           +------------------+
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|                   PHASE 1.3 MULTI-REVIEWER CONSENSUS ENGINE                        |
| - Deterministic Aggregation (Unanimous / Majority / Adjudication Required)        |
| - Statistical Inter-Rater Agreement: Cohen's Kappa (N=2), Fleiss' Kappa (N>=3)    |
| - Binding Dispute Adjudication with Mandatory Clinical Justification               |
| - Consensus Report Synthesis & Append-Only Audit Logging                           |
| - Storage: data/consensus/CXR1122.json                                             |
+------------------------------------------------------------------------------------+
```

> **RESEARCH PROTOTYPE DISCLAIMER**  
> This system is an investigational research prototype for explainability and inter-rater reliability analysis. It is **NOT** a certified medical device, automated clinical diagnostic tool, or ground-truth reference database. Consensus decisions and adjudication records represent research metadata only.

---

## 2. Storage & Directory Model

To ensure strict reviewer isolation and prevent race conditions or unintentional data leakage:

| Artifact Type | Directory / Filename Pattern | Description |
| :--- | :--- | :--- |
| **Machine Evidence** | `data/iu_xray/` | Authoritative immutable DenseNet-121, QA, and grounding artifacts. |
| **Individual Reviews** | `data/reviews/{study_id}_{reviewer_id}.json` | Scoped review session for a specific clinical reviewer (e.g. `CXR1122_dr_alice_01.json`). |
| **Default Review** | `data/reviews/{study_id}.json` | Backward-compatible single-reviewer file (used by default/`researcher_01`). |
| **Consensus Session** | `data/consensus/{study_id}.json` | Central consensus synthesis, statistical metrics, adjudication records, and consensus report draft. |

---

## 3. Deterministic Consensus Aggregation Rules

Given $N$ independent reviewer decisions for a candidate finding or QA question ($N \ge N_{\text{min}}$, default $N_{\text{min}}=2$):

1. **Unanimous Consensus ($100\%$ Agreement)**:
   $$\forall i, j \in \{1, \dots, N\}, \quad d_i = d_j = d^*$$
   - `consensus_status`: `"unanimous"`
   - `consensus_decision`: $d^*$
   - `agreement_ratio`: $1.0$

2. **Simple Majority Consensus ($> 50\%$ Agreement)**:
   $$\text{count}(d^*) > \frac{N}{2}$$
   - `consensus_status`: `"majority"`
   - `consensus_decision`: $d^*$
   - `agreement_ratio`: $\frac{\text{count}(d^*)}{N}$

3. **Disagreement / Tie / `needs_review` ($ \le 50\%$ or Flagged)**:
   $$\text{count}(d_{\text{modal}}) \le \frac{N}{2} \quad \lor \quad \exists i : d_i = \text{"needs\_review"}$$
   - `consensus_status`: `"adjudication_required"`
   - `consensus_decision`: `null` (requires lead adjudicator binding decision)
   - Automatically flagged as a pending item in `session.adjudication.items_requiring_adjudication`.

4. **Independent Location and Severity Consensus**:
   - Location and Severity follow the identical modal rule independently from presence decisions. Disagreements in location or severity trigger independent location/severity adjudication without invalidating presence consensus.

---

## 4. Statistical Inter-Rater Reliability Metrics

The consensus engine evaluates three statistical reliability measures across all candidate findings:

### 4.1. Average Agreement Ratio
$$\text{AAR} = \frac{1}{|F|} \sum_{f \in F} \frac{\max_{c \in C} n_{f, c}}{N}$$
where $|F|$ is the number of candidate findings (7), $N$ is the number of active raters, and $n_{f, c}$ is the count of raters choosing category $c$ for finding $f$.

### 4.2. Cohen's Kappa ($\kappa$) — Exactly 2 Raters ($N = 2$)
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$
- **Observed Agreement ($P_o$)**: $P_o = \frac{1}{|F|} \sum_{f=1}^{|F|} \mathbb{I}(d_{f, 1} == d_{f, 2})$
- **Expected Chance Agreement ($P_e$)**: $P_e = \sum_{c \in C} p_{1, c} \cdot p_{2, c}$
- **Robust Boundary Handling**: If $P_e = 1.0$ and $P_o = 1.0$, $\kappa = 1.0$. If undefined or $|F| = 0$, returns `null` (never `NaN` or `Infinity`). Clamped to $[-1.0, 1.0]$.

### 4.3. Fleiss' Kappa ($\kappa$) — 3 or More Raters ($N \ge 3$)
$$\kappa = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}$$
- **Observed Proportion of Agreement ($\bar{P}$)**:
  $$P_i = \frac{1}{N(N - 1)} \sum_{j=1}^k n_{ij}(n_{ij} - 1) = \frac{1}{N(N - 1)} \left( \sum_{j=1}^k n_{ij}^2 - N \right)$$
  $$\bar{P} = \frac{1}{|F|} \sum_{i=1}^{|F|} P_i$$
- **Expected Proportion by Chance ($\bar{P}_e$)**:
  $$p_j = \frac{1}{|F| N} \sum_{i=1}^{|F|} n_{ij}, \qquad \bar{P}_e = \sum_{j=1}^k p_j^2$$
- **Robustness**: When all raters assign all items to the exact same category ($\bar{P} = 1.0, \bar{P}_e = 1.0$), $\kappa = 1.0$.

### 4.4. Qualitative Interpretation Scale (Landis & Koch, 1977)
| Kappa Range ($\kappa$) | Qualitative Reliability Interpretation | UI Status Badge |
| :--- | :--- | :--- |
| $\ge 0.81$ | Almost Perfect Agreement | `status-supported` (Green) |
| $0.61 - 0.80$ | Substantial Agreement | `status-supported` (Green) |
| $0.41 - 0.60$ | Moderate Agreement | `status-possible` (Amber) |
| $0.21 - 0.40$ | Fair Agreement | `status-neutral` (Gray) |
| $0.00 - 0.20$ | Slight Agreement | `status-neutral` (Gray) |
| $< 0.00$ | Poor / Systematic Disagreement | `status-needs-review` (Rose) |

---

## 5. Dispute Adjudication Workflow

When diagnostic consensus cannot be established by simple majority, or when a reviewer flags a finding as `needs_review`:
1. Item is automatically enqueued in `session.adjudication.items_requiring_adjudication`.
2. Lead Adjudicator inspects the multi-reviewer decision distribution via the dashboard or API.
3. Adjudicator submits a binding decision (`POST /api/studies/{study_id}/consensus/adjudicate`).
4. **Mandatory Justification**: An empty or whitespace reason is rejected with `400 Bad Request`.
5. The adjudication record is logged to `session.adjudication.adjudication_records` and the append-only `audit_trail`.
6. Consensus status transitions to `adjudicated`, resolving the pending item.

---

## 6. REST API Specification

| Route | Method | Payload / Query | Description |
| :--- | :---: | :--- | :--- |
| `/api/studies/{id}/consensus` | `GET` | - | Fetch full consensus session, metrics, findings, and adjudication state. |
| `/api/studies/{id}/consensus` | `POST` | `required_reviewers`, `minimum_reviewers`, `reviewers` | Configure/initialize consensus session cohort. |
| `/api/studies/{id}/consensus/reviewers` | `GET` | - | List all registered reviewers and their completion status. |
| `/api/studies/{id}/consensus/reviewers` | `POST` | `reviewer_id`, `display_name`, `role` | Register new reviewer into study consensus cohort. |
| `/api/studies/{id}/consensus/finding/{name}` | `GET` | - | Query individual finding consensus breakdown & decisions. |
| `/api/studies/{id}/consensus/agreement` | `GET` | - | Query statistical metrics (AAR, Cohen's Kappa, Fleiss' Kappa). |
| `/api/studies/{id}/consensus/adjudicate` | `POST` | `target_type`, `target_id`, `decision`, `adjudicator_id`, `reason` | Submit binding dispute adjudication with mandatory rationale. |
| `/api/studies/{id}/consensus/adjudication` | `GET` | - | Fetch pending items and completed adjudication records. |
| `/api/studies/{id}/consensus/report` | `GET` | - | Fetch draft/finalized consensus report. |
| `/api/studies/{id}/consensus/report` | `POST` | `final_findings`, `final_impression`, `reviewer_comment` | Save customized consensus report draft. |
| `/api/studies/{id}/consensus/finalize` | `POST` | `adjudicator_id`, `reason` | Validate and permanently lock consensus session. |
| `/api/studies/{id}/consensus/audit` | `GET` | - | Fetch append-only audit trail of all consensus actions. |
| `/api/studies/{id}/consensus/export` | `GET` | `?format=json` or `?format=text` | Download comprehensive consensus report package. |

---

## 7. Verification & Test Suite Summary

The multi-reviewer consensus engine has been rigorously verified against 33 dedicated Phase 1.3 test cases and all 135 legacy test cases (168 total passing tests):

```
----------------------------------------------------------------------
Ran 168 tests in 12.516s

OK
```

### Verified Test Categories:
1. **Cohort & Session Management**: Tests 1–5 (Creation, Registration, Idempotence, Isolation, Minimum threshold enforcement).
2. **Deterministic Aggregation**: Tests 6–13 (Aggregation, Unanimous, Majority, Tie detection, Needs-review handling, Location/Severity, QA consensus).
3. **Statistical Agreement & Edge Cases**: Tests 14–20 (Agreement ratio, Cohen's Kappa $N=2$, Fleiss' Kappa $N \ge 3$, Zero-variance safety, Metric bounds $[-1.0, 1.0]$, No NaN/Infinity).
4. **Adjudication Operations**: Tests 21–25 (Mandatory reason enforcement, Binding resolution, State transition, Append-only audit logging).
5. **Safety Invariants & Immutability**: Tests 26–27 (Machine evidence byte immutability, Zero ground-truth XML report leakage).
6. **Robustness & Backward Compatibility**: Tests 28–33 (404 handling, Path traversal rejection, Seamless Phase 1.0/1.1/1.2 coexistence, JSON/Text export, Post-finalization lock).
