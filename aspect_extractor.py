"""
Part 1: Aspect Term Extraction (ATE)
Syntactic Compound Noun Pattern Matching from Scratch
"""

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])

STOP_NOUNS = {"phone", "device", "item", "product", "thing", "day", "time", "amazon"}
OPINION_WORDS = {"good", "great", "excellent", "amazing", "bad", "terrible", "awful", "poor", "nice"}

def extract_aspect_candidates(text: str):
    if not text:
        return []

    doc = nlp(text)
    aspects, seen, n, i = [], set(), len(doc), 0

    while i < n:
        tok = doc[i]
        
        # Rule 1: Compound Noun (e.g. "battery life", "customer service")
        is_compound = tok.pos_ in ["NOUN", "PROPN"] or tok.dep_ == "compound"
        if i + 1 < n and is_compound and doc[i+1].pos_ in ["NOUN", "PROPN"]:
            t1, t2 = tok.text.lower(), doc[i+1].text.lower()
            term = t2 if t1 in STOP_NOUNS else f"{t1} {t2}"
            if term not in seen and term not in STOP_NOUNS:
                aspects.append({"term": term, "start_token": i, "end_token": i + 1})
                seen.add(term)
            i += 2
            continue

        # Rule 2: Feature Adjective + Noun (e.g. "touch screen", "sound quality")
        if i + 1 < n and tok.pos_ == "ADJ" and doc[i+1].pos_ in ["NOUN", "PROPN"]:
            adj, noun = tok.text.lower(), doc[i+1].text.lower()
            if adj not in OPINION_WORDS and noun not in STOP_NOUNS:
                term = f"{adj} {noun}"
                if term not in seen:
                    aspects.append({"term": term, "start_token": i, "end_token": i + 1})
                    seen.add(term)
                i += 2
                continue

        # Rule 3: Standalone Noun (e.g. "camera", "screen", "price")
        if tok.pos_ in ["NOUN", "PROPN"]:
            term = tok.text.lower()
            if len(term) > 2 and term not in STOP_NOUNS and term not in seen:
                aspects.append({"term": term, "start_token": i, "end_token": i})
                seen.add(term)

        i += 1

    return aspects

if __name__ == "__main__":
    sample = "The OLED screen is gorgeous and crisp, but the battery life is completely terrible."
    for asp in extract_aspect_candidates(sample):
        print(f"  Aspect: '{asp['term']}' (Tokens [{asp['start_token']}:{asp['end_token']}])")
