"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.5 — IU X-Ray Report Corpus Analysis for Diagnostic QA Design

Analyzes all 3,955 clinical XML reports in data/iu_xray/reports/ecgen-radiology:
1. Section extraction (FINDINGS, IMPRESSION, INDICATION, COMPARISON).
2. Frequency of 18 TorchXRayVision pathologies (positive vs negated/negative mentions).
3. Frequency of additional common radiological concepts in IU X-Ray (granuloma, scoliosis, calcification, surgical clips, etc.).
4. Empirical availability of Location descriptors (right, left, bilateral, base/basilar, apex/apical, upper/lower/middle lobe, retrocardiac).
5. Empirical availability of Severity/Extent descriptors (mild, moderate, severe, small, large, minimal, trace, subtle).
6. Ground-truth case extraction (normal, single abnormal finding, multiple findings).
"""

import os
import sys
import json
import glob
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


TORCHXRAYVISION_PATHOLOGIES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum"
]

# Regex patterns for clinical concept matching
CONCEPT_PATTERNS = {
    "Atelectasis": r"\b(atelecta(sis|tic)|collapse)\b",
    "Consolidation": r"\b(consolidation|airspace disease|air space disease)\b",
    "Infiltration": r"\b(infiltrat(e|ion|es|ing))\b",
    "Pneumothorax": r"\b(pneumothorax|pneumothoraces)\b",
    "Edema": r"\b(edema|pulmonary edema|vascular congestion|congestion)\b",
    "Emphysema": r"\b(emphysema|hyperinflation|hyperexpansion|copd)\b",
    "Fibrosis": r"\b(fibrosis|fibrotic|interstitial marking(s)?)\b",
    "Effusion": r"\b(effusion|pleural effusion|blunting)\b",
    "Pneumonia": r"\b(pneumonia|infectious process|bronchopneumonia)\b",
    "Pleural_Thickening": r"\b(pleural thickening|apical capping|pleural plaque(s)?)\b",
    "Cardiomegaly": r"\b(cardiomegaly|cardiac enlargement|enlarged heart|prominent cardiac silhouette)\b",
    "Nodule": r"\b(nodule(s)?|granuloma(s)?|nodular density)\b",
    "Mass": r"\b(mass|masses|neoplasm|tumor)\b",
    "Hernia": r"\b(hernia|hiatal hernia|hiatus hernia)\b",
    "Lung Lesion": r"\b(lesion|cavitary lesion|parenchymal lesion)\b",
    "Fracture": r"\b(fracture(s)?|broken rib(s)?|rib fracture(s)?)\b",
    "Lung Opacity": r"\b(opacit(y|ies)|density|densities|haze|streaky opacity)\b",
    "Enlarged Cardiomediastinum": r"\b(enlarged cardiomediastinum|mediastinal widening|prominent mediastinum|enlarged mediastinum)\b"
}

# Non-vision model concepts commonly present in reports
ADDITIONAL_CONCEPTS = {
    "Normal / Clear Lungs": r"\b(no (acute )?abnormality|lungs are clear|clear of (focal )?infiltrate|unremarkable|within normal limits|normal chest)\b",
    "Calcified Granuloma / Calcification": r"\b(calcifi(ed|cation|cations)|calcified granuloma(s)?)\b",
    "Degenerative Spine / Scoliosis / Bone Disease": r"\b(scoliosis|degenerative|spondylosis|osteophyte(s)?|kyphosis|disc disease)\b",
    "Aortic Tortuosity / Atherosclerosis": r"\b(tortuous aorta|atherosclerosis|aortic ectasia|aortic calcification)\b",
    "Support Devices / Hardware / Post-Surgical": r"\b(sternotomy|clips|pacemaker|leads|catheter|stent|hardware|arthroplasty)\b",
    "Granulomatous Disease": r"\b(granulomatous disease|histoplasmosis)\b"
}

LOCATION_PATTERNS = {
    "Right": r"\b(right|right-sided|rt)\b",
    "Left": r"\b(left|left-sided|lt)\b",
    "Bilateral": r"\b(bilateral|both lungs|both bases|diffuse)\b",
    "Upper Lobe / Apical": r"\b(upper lobe|apical|apex|apices|upper lung)\b",
    "Middle Lobe / Lingula": r"\b(middle lobe|rml|lingula)\b",
    "Lower Lobe / Basilar / Base": r"\b(lower lobe|basilar|base|bases|retrocardiac|costophrenic)\b",
    "Mediastinum / Hilum": r"\b(mediastin(um|al)|hilar|hilum|perihilar)\b"
}

SEVERITY_PATTERNS = {
    "Small / Minimal / Trace": r"\b(small|minimal|trace|tiny|subtle|mild|minor)\b",
    "Moderate": r"\b(moderate|modest)\b",
    "Large / Severe": r"\b(large|severe|gross|marked|extensive|dense)\b"
}

NEGATION_PATTERNS = r"\b(no|without|negative for|free of|clear of|no evidence of|no sign of|denies|absence of)\b"


def analyze_corpus(reports_dir: str):
    xml_files = glob.glob(os.path.join(reports_dir, "*.xml"))
    total_reports = len(xml_files)
    print(f"Loaded {total_reports} XML reports from {reports_dir}")

    total_with_findings = 0
    total_with_impression = 0
    total_with_both = 0
    total_with_indication = 0
    total_with_comparison = 0

    pathology_positive_counts = Counter()
    pathology_negated_counts = Counter()
    additional_concept_counts = Counter()

    location_counts = Counter()
    severity_counts = Counter()

    findings_word_counts = []
    impression_word_counts = []

    case_studies = {
        "normal": [],
        "single_finding": [],
        "multiple_findings": []
    }

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            uid_node = root.find("uId")
            study_id = uid_node.attrib.get("id") if uid_node is not None else os.path.splitext(os.path.basename(xml_path))[0]

            findings = ""
            impression = ""
            indication = ""
            comparison = ""

            for abs_text in root.iter("AbstractText"):
                label = (abs_text.attrib.get("Label") or "").upper().strip()
                text = (abs_text.text or "").strip()
                if label == "FINDINGS":
                    findings = text
                elif label == "IMPRESSION":
                    impression = text
                elif label == "INDICATION":
                    indication = text
                elif label == "COMPARISON":
                    comparison = text

            has_findings = bool(findings)
            has_impression = bool(impression)

            if has_findings:
                total_with_findings += 1
                findings_word_counts.append(len(findings.split()))
            if has_impression:
                total_with_impression += 1
                impression_word_counts.append(len(impression.split()))
            if has_findings and has_impression:
                total_with_both += 1
            if indication:
                total_with_indication += 1
            if comparison:
                total_with_comparison += 1

            full_text = f"{findings} {impression}".lower()

            # Analyze 18 TorchXRayVision pathologies in text
            active_positive_findings_in_report = []

            for path_name, pat in CONCEPT_PATTERNS.items():
                matches = list(re.finditer(pat, full_text, re.IGNORECASE))
                if matches:
                    # Check if negated
                    is_pos = False
                    for m in matches:
                        start_idx = max(0, m.start() - 40)
                        context_window = full_text[start_idx:m.end()]
                        if re.search(NEGATION_PATTERNS, context_window, re.IGNORECASE):
                            pathology_negated_counts[path_name] += 1
                        else:
                            pathology_positive_counts[path_name] += 1
                            is_pos = True
                    if is_pos:
                        active_positive_findings_in_report.append(path_name)

            # Analyze additional common non-model concepts
            for c_name, pat in ADDITIONAL_CONCEPTS.items():
                if re.search(pat, full_text, re.IGNORECASE):
                    additional_concept_counts[c_name] += 1

            # Analyze location descriptors
            for loc_name, pat in LOCATION_PATTERNS.items():
                if re.search(pat, full_text, re.IGNORECASE):
                    location_counts[loc_name] += 1

            # Analyze severity descriptors
            for sev_name, pat in SEVERITY_PATTERNS.items():
                if re.search(pat, full_text, re.IGNORECASE):
                    severity_counts[sev_name] += 1

            # Select sample cases for design demonstration
            is_normal_report = "normal" in impression.lower() or "no acute" in impression.lower()
            if is_normal_report and len(active_positive_findings_in_report) == 0 and len(case_studies["normal"]) < 3:
                case_studies["normal"].append({
                    "study_id": study_id,
                    "findings": findings,
                    "impression": impression
                })
            elif len(active_positive_findings_in_report) == 1 and len(case_studies["single_finding"]) < 3:
                case_studies["single_finding"].append({
                    "study_id": study_id,
                    "finding": active_positive_findings_in_report[0],
                    "findings": findings,
                    "impression": impression
                })
            elif len(active_positive_findings_in_report) >= 2 and len(case_studies["multiple_findings"]) < 3:
                case_studies["multiple_findings"].append({
                    "study_id": study_id,
                    "findings_list": active_positive_findings_in_report,
                    "findings": findings,
                    "impression": impression
                })

        except Exception as e:
            print(f"[WARN] Error parsing report {xml_path}: {e}")

    results = {
        "total_reports": total_reports,
        "total_with_findings": total_with_findings,
        "total_with_impression": total_with_impression,
        "total_with_both": total_with_both,
        "total_with_indication": total_with_indication,
        "total_with_comparison": total_with_comparison,
        "avg_findings_words": sum(findings_word_counts) / len(findings_word_counts) if findings_word_counts else 0,
        "avg_impression_words": sum(impression_word_counts) / len(impression_word_counts) if impression_word_counts else 0,
        "pathology_positive_counts": dict(pathology_positive_counts.most_common()),
        "pathology_negated_counts": dict(pathology_negated_counts.most_common()),
        "additional_concept_counts": dict(additional_concept_counts.most_common()),
        "location_counts": dict(location_counts.most_common()),
        "severity_counts": dict(severity_counts.most_common()),
        "case_studies": case_studies
    }

    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "data", "iu_xray", "reports", "ecgen-radiology")

    print("=" * 65)
    print("PHASE 0.5 — IU X-RAY REPORT CORPUS ANALYSIS FOR DIAGNOSTIC QA")
    print("=" * 65)
    print(f"Target Directory: {reports_dir}\n")

    if not os.path.exists(reports_dir):
        print(f"[ERROR] Reports directory not found at {reports_dir}")
        sys.exit(1)

    analysis_results = analyze_corpus(reports_dir)

    print("\n" + "=" * 65)
    print("1. REPORT CORPUS OVERVIEW")
    print("=" * 65)
    print(f"Total Reports Analyzed               : {analysis_results['total_reports']}")
    print(f"Reports with FINDINGS                : {analysis_results['total_with_findings']} ({analysis_results['total_with_findings']/analysis_results['total_reports']*100:.1f}%)")
    print(f"Reports with IMPRESSION              : {analysis_results['total_with_impression']} ({analysis_results['total_with_impression']/analysis_results['total_reports']*100:.1f}%)")
    print(f"Reports with Both Sections           : {analysis_results['total_with_both']} ({analysis_results['total_with_both']/analysis_results['total_reports']*100:.1f}%)")
    print(f"Reports with INDICATION              : {analysis_results['total_with_indication']}")
    print(f"Reports with COMPARISON              : {analysis_results['total_with_comparison']}")
    print(f"Average Words in FINDINGS            : {analysis_results['avg_findings_words']:.1f}")
    print(f"Average Words in IMPRESSION          : {analysis_results['avg_impression_words']:.1f}")

    print("\n" + "=" * 65)
    print("2. TORCHXRAYVISION PATHOLOGIES (POSITIVE vs NEGATED)")
    print("=" * 65)
    print(f"{'Pathology':<28} | {'Positive Mentions':<18} | {'Negated Mentions':<18}")
    print("-" * 65)
    for path in TORCHXRAYVISION_PATHOLOGIES:
        pos = analysis_results['pathology_positive_counts'].get(path, 0)
        neg = analysis_results['pathology_negated_counts'].get(path, 0)
        print(f"{path:<28} | {pos:<18} | {neg:<18}")

    print("\n" + "=" * 65)
    print("3. ADDITIONAL FREQUENT NON-MODEL RADIOLOGICAL CONCEPTS")
    print("=" * 65)
    for concept, count in analysis_results['additional_concept_counts'].items():
        print(f"{concept:<46}: {count} reports ({count/analysis_results['total_reports']*100:.1f}%)")

    print("\n" + "=" * 65)
    print("4. LOCATION DESCRIPTOR AVAILABILITY")
    print("=" * 65)
    for loc, count in analysis_results['location_counts'].items():
        print(f"{loc:<30}: {count} reports ({count/analysis_results['total_reports']*100:.1f}%)")

    print("\n" + "=" * 65)
    print("5. SEVERITY DESCRIPTOR AVAILABILITY")
    print("=" * 65)
    for sev, count in analysis_results['severity_counts'].items():
        print(f"{sev:<30}: {count} reports ({count/analysis_results['total_reports']*100:.1f}%)")

    # Save to JSON
    output_json_path = os.path.join(base_dir, "data", "iu_xray", "corpus_analysis_summary.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"\nSaved analysis summary to: {output_json_path}")


if __name__ == "__main__":
    main()
