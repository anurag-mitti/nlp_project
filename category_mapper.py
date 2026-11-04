"""
Part 2: Semantic Category Mapping (ACD)
Vector Math (Dot Product, L2 Norm, Cosine Similarity) from Scratch
"""

import math

def dot_product(u, v):
    return sum(a * b for a, b in zip(u, v))

def vector_magnitude(u):
    return math.sqrt(dot_product(u, u))

def cosine_similarity(u, v):
    mag = vector_magnitude(u) * vector_magnitude(v)
    return dot_product(u, v) / mag if mag else 0.0

def text_to_vector(text):
    t = text.lower()
    return [float(t.count(ch)) for ch in "abcdefghijklmnopqrstuvwxyz"]

CATEGORIES = {
    "CAMERA": ["camera", "picture", "photo", "lens", "video", "selfie"],
    "BATTERY": ["battery", "charge", "charger", "power", "drain", "backup"],
    "SCREEN": ["screen", "display", "glass", "oled", "touch", "panel"],
    "PERFORMANCE": ["performance", "speed", "ram", "processor", "lag", "gaming", "cpu", "sensor"],
    "BUILD_DESIGN": ["design", "size", "weight", "body", "look", "build"],
    "AUDIO_CALL": ["microphone", "speaker", "sound", "volume", "mic", "call", "voice"],
    "PRICE_VALUE": ["price", "cost", "money", "value", "expensive", "deal"],
    "SERVICE_DELIVERY": ["service", "delivery", "shipping", "seller", "support"]
}

def map_aspect_to_category(term: str):
    if not term:
        return ("GENERAL", 0.0)

    # 1. Direct Anchor Match
    words = term.lower().split()
    for cat, anchors in CATEGORIES.items():
        if any(w in anchors for w in words):
            return (cat, 1.0)

    # 2. Scratch Cosine Similarity over Character Vectors
    term_vec = text_to_vector(term)
    best_cat, best_sim = "GENERAL", 0.0

    for cat, anchors in CATEGORIES.items():
        sims = [cosine_similarity(term_vec, text_to_vector(a)) for a in anchors]
        max_sim = max(sims) if sims else 0.0
        if max_sim > best_sim:
            best_sim, best_cat = max_sim, cat

    return (best_cat, round(max(0.50, min(0.95, best_sim)), 2))

if __name__ == "__main__":
    for word in ["oled screen", "battery life", "camera quality", "speaker", "processor"]:
        cat, conf = map_aspect_to_category(word)
        print(f"  '{word}' -> {cat} (Similarity: {conf})")
