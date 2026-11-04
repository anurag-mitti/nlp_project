"""
Aspect-Based Sentiment Analysis (ABSA) Unified Pipeline
Coordinates the 4 individual components into a single end-to-end interface.
"""

import pandas as pd
from aspect_binder_eval import bind_aspects_and_sentiments

def analyze_review(text: str, window_size: int = 5):
    if not text or not text.strip():
        return {"text": text, "aspect_count": 0, "aspects": []}

    results = bind_aspects_and_sentiments(text, window_size=window_size)
    return {"text": text, "aspect_count": len(results), "aspects": results}

def analyze_corpus(texts, window_size: int = 5):
    records = []
    for idx, text in enumerate(texts):
        for asp in bind_aspects_and_sentiments(text, window_size=window_size):
            records.append({"review_id": idx, "text": text, **asp})
    return pd.DataFrame(records)

def get_category_summary(df_aspects: pd.DataFrame):
    if df_aspects.empty:
        return pd.DataFrame()
    summary = df_aspects.groupby(["category", "polarity"]).size().unstack(fill_value=0)
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["Total"] = summary["Positive"] + summary["Negative"] + summary["Neutral"]
    summary["Positive_Pct"] = (summary["Positive"] / summary["Total"] * 100).round(1)
    return summary.sort_values(by="Total", ascending=False)

if __name__ == "__main__":
    test = "The camera takes stunning sharp photos, but the customer service was awful and battery drains fast."
    res = analyze_review(test)
    for a in res["aspects"]:
        print(f"  [{a['category']}] '{a['aspect']}' -> {a['polarity']} ({a['confidence']*100:.0f}%) | Context: '{a['context_window']}'")
