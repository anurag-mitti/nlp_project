"""
Part 3: Sentiment Classification Engine (APC)
Lexicon Scoring, Negation Reversal, and Valence Scaling from Scratch
"""

import math
import re

POSITIVE_WORDS = {
    "great": 1.5, "good": 1.0, "excellent": 2.0, "amazing": 2.0, "awesome": 2.0,
    "stunning": 1.8, "gorgeous": 1.8, "crisp": 1.4, "fast": 1.3, "smooth": 1.4,
    "perfect": 2.0, "best": 1.8, "clear": 1.2, "solid": 1.3, "work": 1.0, "works": 1.0,
    "lasts": 1.4, "reasonable": 1.3, "affordable": 1.3
}

NEGATIVE_WORDS = {
    "terrible": -2.0, "horrible": -2.0, "awful": -2.0, "bad": -1.2, "poor": -1.4,
    "worst": -2.0, "slow": -1.3, "lag": -1.4, "drain": -1.3, "flimsy": -1.5,
    "cheap": -1.1, "expensive": -1.3, "broken": -1.8, "freezes": -1.5, "fails": -1.5
}

NEGATIONS = {"not", "never", "no", "isnt", "wasnt", "doesnt", "didnt", "cant", "wont", "dont", "rarely"}
INTENSIFIERS = {"very": 1.5, "extremely": 1.8, "really": 1.4, "completely": 1.6, "too": 1.3}
DIMINISHERS = {"slightly": 0.7, "somewhat": 0.7, "bit": 0.8}

def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))

def score_token_sequence(tokens):
    clean = [re.sub(r"[^a-z]", "", t.lower()) for t in tokens if t]
    clean = [t for t in clean if t]

    total_score, matched = 0.0, 0
    for i, tok in enumerate(clean):
        base = POSITIVE_WORDS.get(tok, NEGATIVE_WORDS.get(tok, 0.0))
        if base:
            matched += 1
            lookback = clean[max(0, i - 3):i]
            mult = 1.0
            for w in lookback:
                mult *= INTENSIFIERS.get(w, DIMINISHERS.get(w, 1.0))
            score = base * mult * (-0.95 if any(w in NEGATIONS for w in lookback) else 1.0)
            total_score += score

    if not matched:
        return ("Neutral", 0.0, 0.50)

    avg = total_score / matched
    polarity = "Positive" if avg > 0.20 else ("Negative" if avg < -0.20 else "Neutral")
    confidence = round(0.65 + (sigmoid(abs(avg)) - 0.50) * 0.65, 2)
    return (polarity, round(avg, 2), min(0.99, confidence))

if __name__ == "__main__":
    tests = ["is gorgeous and crisp", "is completely terrible", "does not work", "rarely works"]
    for t in tests:
        pol, score, conf = score_token_sequence(t.split())
        print(f"  '{t}' -> {pol} (Score: {score}, Conf: {conf})")
