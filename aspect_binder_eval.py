"""
Part 4: Aspect-Sentiment Binding & Evaluation Metrics
Contextual Window Boundary Clipping & Pure-Python Metric Suite
"""

import spacy
from aspect_extractor import extract_aspect_candidates
from category_mapper import map_aspect_to_category
from sentiment_engine import score_token_sequence

nlp = spacy.load("en_core_web_sm", disable=["ner"])
CONTRAST_WORDS = {"but", "however", "although", "though", "yet", ";", "."}

# 1. Window-Based Aspect-Sentiment Binding with Boundary Protection
def bind_aspects_and_sentiments(text: str, window_size: int = 5):
    if not text:
        return []

    tokens = [tok.text for tok in nlp(text)]
    results = []

    for cand in extract_aspect_candidates(text):
        term = cand["term"]
        s_tok, e_tok = cand["start_token"], cand["end_token"]
        category, _ = map_aspect_to_category(term)

        # Context Window (+- window_size tokens) with Contrast Boundary Protection
        win_start = max(0, s_tok - window_size)
        win_end = min(len(tokens), e_tok + window_size + 1)

        for idx in range(s_tok - 1, win_start - 1, -1):
            if tokens[idx].lower() in CONTRAST_WORDS:
                win_start = idx + 1
                break

        for idx in range(e_tok + 1, win_end):
            if tokens[idx].lower() in CONTRAST_WORDS:
                win_end = idx
                break

        context = tokens[win_start:win_end]
        polarity, score, conf = score_token_sequence(context)

        results.append({
            "aspect": term,
            "category": category,
            "polarity": polarity,
            "sentiment_score": score,
            "confidence": conf,
            "context_window": " ".join(context)
        })

    return results

# 2. Evaluation Metrics Implemented from Scratch
def calculate_metrics(y_true, y_pred, labels=None):
    if not y_true or len(y_true) != len(y_pred):
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": []}

    labels = labels or sorted(list(set(y_true) | set(y_pred)))
    cm = [[sum(yt == r and yp == c for yt, yp in zip(y_true, y_pred)) for c in labels] for r in labels]
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

    f1_scores = []
    for l in labels:
        tp = sum(yt == l and yp == l for yt, yp in zip(y_true, y_pred))
        fp = sum(yt != l and yp == l for yt, yp in zip(y_true, y_pred))
        fn = sum(yt == l and yp != l for yt, yp in zip(y_true, y_pred))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append((2 * p * r) / (p + r) if (p + r) else 0.0)

    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(sum(f1_scores) / len(labels) if labels else 0.0, 3),
        "confusion_matrix": cm,
        "labels": labels
    }

if __name__ == "__main__":
    sample = "The OLED screen is gorgeous and crisp, but the battery life is completely terrible."
    for b in bind_aspects_and_sentiments(sample):
        print(f"  [{b['category']}] '{b['aspect']}' -> {b['polarity']} (Window: '{b['context_window']}')")
