import os
from typing import List, Tuple, Dict, Any, Optional
from pyabsa import AspectTermExtraction as ATEPC

_CACHED_EXTRACTOR = None

def load_aspect_extractor(checkpoint: str = "english", auto_device: bool = True) -> ATEPC.AspectExtractor:
    global _CACHED_EXTRACTOR
    if _CACHED_EXTRACTOR is not None:
        return _CACHED_EXTRACTOR

    local_checkpoint_dir = os.path.join("checkpoints", "ATEPC_ENGLISH_CHECKPOINT")
    target_cp = checkpoint
    if checkpoint == "english" and os.path.exists(local_checkpoint_dir):
        target_cp = "english"

    extractor = ATEPC.AspectExtractor(checkpoint=target_cp, auto_device=auto_device)
    _CACHED_EXTRACTOR = extractor
    return extractor

def extract_aspect_sentiments(text: str, extractor: Optional[ATEPC.AspectExtractor] = None) -> List[Tuple[str, str, float]]:
    if not text or not text.strip():
        return []

    if extractor is None:
        extractor = load_aspect_extractor()

    try:
        prediction = extractor.predict(text.strip(), save_result=False, print_result=False)
    except Exception:
        return []

    if isinstance(prediction, list):
        if not prediction:
            return []
        prediction = prediction[0]

    aspects = prediction.get("aspect", [])
    sentiments = prediction.get("sentiment", [])
    confidences = prediction.get("confidence", [])

    results = []
    for aspect, sentiment, conf in zip(aspects, sentiments, confidences):
        results.append((str(aspect).strip(), str(sentiment).capitalize(), round(float(conf), 4)))

    return results

def batch_extract_aspect_sentiments(texts: List[str], extractor: Optional[ATEPC.AspectExtractor] = None, progress_callback=None) -> List[Dict[str, Any]]:
    if extractor is None:
        extractor = load_aspect_extractor()

    batch_results = []
    total = len(texts)

    for i, text in enumerate(texts):
        if progress_callback:
            progress_callback(i + 1, total)

        if not isinstance(text, str) or not text.strip():
            batch_results.append({"text": text, "aspects": []})
            continue

        aspects = extract_aspect_sentiments(text, extractor=extractor)
        batch_results.append({"text": text, "aspects": aspects})

    return batch_results
