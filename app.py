"""
Aspect-Based Sentiment Analysis (ABSA) Streamlit Application
4-Stage Modular NLP Pipeline Built from Scratch
"""

import streamlit as st
import pandas as pd
import os

from aspect_extractor import extract_aspect_candidates
from category_mapper import map_aspect_to_category
from sentiment_engine import score_token_sequence
from aspect_binder_eval import bind_aspects_and_sentiments
from pipeline import analyze_review, analyze_corpus, get_category_summary

st.set_page_config(page_title="Aspect-Based Sentiment Analysis", layout="wide")

st.markdown("""
<style>
    .badge-pos { background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    .badge-neg { background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    .badge-neu { background-color: #e2e3e5; color: #383d41; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Pipeline Architecture")
st.sidebar.markdown("""
**Core Modules Built from Scratch:**
1. **Stage 1:** Syntactic Aspect Extraction (`aspect_extractor.py`)
2. **Stage 2:** Semantic Category Mapping (`category_mapper.py`)
3. **Stage 3:** Valence-Shifter Sentiment Engine (`sentiment_engine.py`)
4. **Stage 4:** Context Window Binding & Evaluation (`aspect_binder_eval.py`)
""")
window_param = st.sidebar.slider("Context Window Size (tokens)", 2, 8, 5)

# Main Page
st.title("Aspect-Based Sentiment Analysis (ABSA)")
st.caption("A granular NLP pipeline extracting product aspects, categories, and isolated sentiments.")

tab1, tab2 = st.tabs(["1. Single Review Inspector", "2. Batch Corpus Analytics"])

with tab1:
    st.subheader("Interactive Review Inspector")
    presets = [
        "The OLED screen is gorgeous and crisp, but the battery life is completely terrible.",
        "The camera takes stunning sharp photos, but the customer service was awful and battery drains fast.",
        "Solid build quality and great performance for gaming, though the price is expensive.",
        "Sound quality is clear during calls, but the touchscreen sometimes freezes.",
        "Fast processor with smooth gaming performance and no lag whatsoever.",
        "The battery charges very fast, but the plastic build feels cheap and flimsy.",
        "The speaker sound is loud and clear, but the fingerprint sensor does not work.",
        "Great value for money and a reasonable price for such a premium display.",
        "Front camera takes blurry photos, but the battery lasts two full days.",
        "Terrible customer service and slow delivery, but the product works fine."
    ]

    csv_path = "data/sample_reviews.csv"
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = presets[0]

    col_select, col_rand = st.columns([3, 1])
    with col_select:
        selected_text = st.selectbox("Select Sample Preset Review:", presets, index=0)
        # Update text if user changed preset
        if st.session_state.get("last_selected") != selected_text:
            st.session_state["input_text"] = selected_text
            st.session_state["last_selected"] = selected_text

    with col_rand:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random Dataset Review"):
            if os.path.exists(csv_path):
                df_sample = pd.read_csv(csv_path)
                random_body = df_sample["body"].dropna().sample(1).iloc[0].strip()
                st.session_state["input_text"] = random_body
                st.rerun()

    review_input = st.text_area("Customer Review (select preset, load random, or type custom):", value=st.session_state["input_text"], height=85)

    if st.button("Run ABSA Pipeline", type="primary"):
        if review_input.strip():
            results = bind_aspects_and_sentiments(review_input, window_size=window_param)
            
            if not results:
                st.info("No aspect terms identified in this sentence.")
            else:
                st.markdown("### Bound Aspect-Sentiment Results")
                st.dataframe(pd.DataFrame(results)[["aspect", "category", "polarity", "confidence", "sentiment_score", "context_window"]], width='stretch')

                st.markdown("---")
                st.markdown("### Step-by-Step Module Inspection")
                
                # Stage 1
                with st.expander("Stage 1: Aspect Term Extraction (Syntactic Chunker)", expanded=True):
                    cands = extract_aspect_candidates(review_input)
                    st.dataframe(pd.DataFrame(cands), width='stretch')

                # Stage 2
                with st.expander("Stage 2: Category Mapping (Vector Cosine Math)", expanded=True):
                    cat_rows = [{"Aspect": c["term"], "Category": map_aspect_to_category(c["term"])[0], "Similarity": map_aspect_to_category(c["term"])[1]} for c in cands]
                    st.dataframe(pd.DataFrame(cat_rows), width='stretch')

                # Stage 3 & 4
                with st.expander("Stage 3 & 4: Context Window Binding & Sentiment Resolution", expanded=True):
                    for r in results:
                        badge = "badge-pos" if r["polarity"] == "Positive" else ("badge-neg" if r["polarity"] == "Negative" else "badge-neu")
                        st.markdown(f"**Aspect:** `{r['aspect']}` | **Category:** `{r['category']}`")
                        st.markdown(f"- Context: `\"{r['context_window']}\"`")
                        st.markdown(f"- Sentiment: <span class='{badge}'>{r['polarity']}</span> (Score: `{r['sentiment_score']}`, Conf: `{r['confidence']*100:.0f}%`)", unsafe_allow_html=True)
                        st.markdown("")

with tab2:
    st.subheader("Batch Corpus Analytics")
    csv_path = "data/sample_reviews.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        batch_size = st.slider("Select number of reviews to analyze:", 10, min(200, len(df_raw)), 40, step=10)
        
        if st.button("Analyze Batch Corpus"):
            with st.spinner("Processing batch through pipeline..."):
                texts = df_raw["body"].dropna().head(batch_size).tolist()
                df_aspects = analyze_corpus(texts, window_size=window_param)

            if not df_aspects.empty:
                st.success(f"Extracted {len(df_aspects)} aspect mentions across {batch_size} customer reviews.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Aspects", len(df_aspects))
                col2.metric("Positive", len(df_aspects[df_aspects["polarity"] == "Positive"]))
                col3.metric("Negative", len(df_aspects[df_aspects["polarity"] == "Negative"]))

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("#### Mentions by Category")
                    st.bar_chart(df_aspects["category"].value_counts())
                with col_c2:
                    st.markdown("#### Sentiment Distribution")
                    st.dataframe(get_category_summary(df_aspects)[["Positive", "Negative", "Neutral", "Total", "Positive_Pct"]], width='stretch')
            else:
                st.warning("No aspects extracted from the selected batch.")
    else:
        st.info("Run prepare_data.py to create sample reviews.")
