import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from nlp_engine import load_aspect_extractor, extract_aspect_sentiments

def load_semeval_data(xml_file: str, max_sentences: int = 100) -> List[Tuple[str, Dict[str, str]]]:
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"SemEval file not found: {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    samples = []
    for sent in root.findall("sentence"):
        text_node = sent.find("text")
        if text_node is None or not text_node.text:
            continue
        text = text_node.text.strip()

        aspect_terms = sent.find("aspectTerms")
        gt_aspects = {}
        if aspect_terms is not None:
            for a in aspect_terms.findall("aspectTerm"):
                term = a.get("term", "").strip().lower()
                polarity = a.get("polarity", "").strip().lower()
                if term and polarity in ["positive", "negative", "neutral"]:
                    gt_aspects[term] = polarity

        if gt_aspects:
            samples.append((text, gt_aspects))
            if len(samples) >= max_sentences:
                break

    return samples

def run_evaluation(xml_path: str = "data/semeval2014/Restaurants_Train_v2.xml", sample_size: int = 60):
    print("======================================================================")
    print("NLP BENCHMARK EVALUATION: PyABSA on SemEval-2014 Task 4")
    print("======================================================================")
    
    print("\nHugging Face datasets command:")
    print("from datasets import load_dataset")
    print("semeval_ds = load_dataset('semeval_2014_task_1_and_4')")

    print(f"\nParsing SemEval dataset: {xml_path} ({sample_size} sentences)...")
    dataset = load_semeval_data(xml_path, max_sentences=sample_size)

    print("\nLoading PyABSA AspectExtractor model (DeBERTa-v3)...")
    extractor = load_aspect_extractor(checkpoint="english")

    print("\nExecuting inference for ATE and APC subtasks...")
    y_true = []
    y_pred = []
    tp, fp, fn = 0, 0, 0

    for idx, (text, gt_map) in enumerate(dataset):
        preds = extract_aspect_sentiments(text, extractor=extractor)
        pred_map = {asp.lower(): sent.lower() for asp, sent, _ in preds}

        for gt_term, gt_pol in gt_map.items():
            matched_pred = None
            for p_term in pred_map:
                if p_term == gt_term or p_term in gt_term or gt_term in p_term:
                    matched_pred = p_term
                    break

            if matched_pred is not None:
                tp += 1
                y_true.append(gt_pol)
                y_pred.append(pred_map[matched_pred])
            else:
                fn += 1

        for p_term in pred_map:
            matched_gt = any(p_term == g or p_term in g or g in p_term for g in gt_map)
            if not matched_gt:
                fp += 1

    ate_precision = tp / (tp + fp) if (tp + fp) else 0.0
    ate_recall = tp / (tp + fn) if (tp + fn) else 0.0
    ate_f1 = (2 * ate_precision * ate_recall) / (ate_precision + ate_recall) if (ate_precision + ate_recall) else 0.0

    apc_accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n======================================================================")
    print("NLP EVALUATION METRICS REPORT")
    print("======================================================================")
    print(f"Evaluated Sentences           : {len(dataset)}")
    print(f"Ground Truth Aspect Spans     : {tp + fn}")
    print(f"Correctly Extracted (TP)      : {tp}")
    print(f"Spurious Extractions (FP)     : {fp}")
    print(f"Missed Aspect Spans (FN)      : {fn}")
    print("----------------------------------------------------------------------")
    print(f"Aspect Term Extraction (ATE) Precision: {ate_precision:.4f} ({ate_precision * 100:.2f}%)")
    print(f"Aspect Term Extraction (ATE) Recall   : {ate_recall:.4f} ({ate_recall * 100:.2f}%)")
    print(f"Aspect Term Extraction (ATE) F1-Score : {ate_f1:.4f} ({ate_f1 * 100:.2f}%)")
    print("----------------------------------------------------------------------")
    print(f"Aspect Polarity Classification (APC) Accuracy   : {apc_accuracy:.4f} ({apc_accuracy * 100:.2f}%)")
    print(f"Aspect Polarity Classification (APC) Macro-F1   : {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print(f"Aspect Polarity Classification (APC) Weighted-F1: {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")
    print("----------------------------------------------------------------------")

    labels = ["positive", "negative", "neutral"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(f"Labels: {labels}")
    print(cm)

    results_summary = {
        "dataset": os.path.basename(xml_path),
        "sample_size": len(dataset),
        "ate_metrics": {
            "precision": round(ate_precision, 4),
            "recall": round(ate_recall, 4),
            "f1_score": round(ate_f1, 4)
        },
        "apc_metrics": {
            "accuracy": round(apc_accuracy, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4)
        },
        "labels": labels,
        "confusion_matrix": cm.tolist()
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=4)
    return results_summary

if __name__ == "__main__":
    run_evaluation()
