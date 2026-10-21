import os
import streamlit as st
import pandas as pd
import plotly.express as px
from nlp_engine import load_aspect_extractor, extract_aspect_sentiments

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }
    .badge-positive {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 4px;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .badge-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 4px;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 4px;
        border: 1px solid #CBD5E1;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading DeBERTa model checkpoint...")
def get_cached_extractor():
    return load_aspect_extractor(checkpoint="english")

try:
    extractor = get_cached_extractor()
except Exception as e:
    st.error(f"Model loading failed: {e}")

with st.sidebar:
    st.title("NLP Project Studio")
    st.caption("Module V: Natural Language Processing")
    
    navigation = st.radio(
        "Navigation",
        options=["Single Review Analysis", "Batch Dataset Analysis"],
        index=0
    )
    
    st.divider()
    st.markdown("### How the Model Works")
    st.markdown("""
    - **Language Model:** DeBERTa-v3 (understands word context)
    - **Step 1:** Finds specific features being reviewed (e.g., screen, battery)
    - **Step 2:** Zooms in on the words around each feature (Local Context Focus)
    - **Step 3:** Classifies sentiment for each feature (Positive, Negative, Neutral)
    - **Benchmark Dataset:** SemEval-2014 Task 4
    """)
    st.divider()
    st.caption("Academic Sem-5 NLP Project")

if navigation == "Single Review Analysis":
    st.markdown('<div class="main-header">Single Review Aspect Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Extract specific features from a review and check how the customer feels about each one.</div>',
        unsafe_allow_html=True
    )

    st.write("##### Sample Reviews (Click to Test):")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    preset_text = ""
    
    with col_ex1:
        if st.button("Sample 1: Phone Review", use_container_width=True):
            preset_text = "The OLED screen is gorgeous and the battery lasts two full days, but the camera app constantly lags and the back glass scratches easily."
    with col_ex2:
        if st.button("Sample 2: Restaurant Review", use_container_width=True):
            preset_text = "The pasta and garlic bread were absolutely mouthwatering, but the waiters were inattentive and the wait time was ridiculous."
    with col_ex3:
        if st.button("Sample 3: Laptop Review", use_container_width=True):
            preset_text = "Fast boot time and the keyboard feel is top notch, however the fan noise under heavy load is unbearable."

    default_value = preset_text if preset_text else "The battery life is amazing and the display is super crisp, but the speaker quality is disappointing."
    
    review_input = st.text_area(
        "Enter customer review text:",
        value=default_value,
        height=120,
        placeholder="Type or paste a customer review here..."
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked or preset_text:
        if not review_input.strip():
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Analyzing review and extracting features..."):
                results = extract_aspect_sentiments(review_input, extractor=extractor)

            if not results:
                st.info("No specific features were detected in the input review.")
            else:
                pos_count = sum(1 for _, s, _ in results if s == "Positive")
                neg_count = sum(1 for _, s, _ in results if s == "Negative")
                neu_count = sum(1 for _, s, _ in results if s == "Neutral")
                
                st.write("---")
                st.subheader("Summary")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Features Found", len(results))
                with m2:
                    st.metric("Positive Features", pos_count)
                with m3:
                    st.metric("Negative Features", neg_count)
                with m4:
                    st.metric("Neutral Features", neu_count)

                st.write("---")
                st.subheader("Extracted Features & Sentiments")

                cols = st.columns(min(len(results), 4))
                for idx, (aspect, sentiment, conf) in enumerate(results):
                    col = cols[idx % len(cols)]
                    with col:
                        if sentiment == "Positive":
                            badge_cls = "badge-positive"
                            border_color = "#16A34A"
                        elif sentiment == "Negative":
                            badge_cls = "badge-negative"
                            border_color = "#DC2626"
                        else:
                            badge_cls = "badge-neutral"
                            border_color = "#64748B"

                        st.markdown(f"""
                        <div style="border-left: 4px solid {border_color}; background: #F8FAFC; padding: 14px; border-radius: 6px; margin-bottom: 12px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
                            <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Feature / Aspect</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 4px 0 8px 0;">{aspect}</div>
                            <span class="{badge_cls}">{sentiment.upper()}</span>
                            <div style="font-size: 0.8rem; color: #64748B; margin-top: 8px;">Confidence: <b>{conf * 100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                if len(results) > 1:
                    df_single = pd.DataFrame(results, columns=["Aspect", "Sentiment", "Confidence"])
                    fig_single = px.pie(
                        df_single,
                        names="Sentiment",
                        title="Sentiment Ratio (Single Review)",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#16A34A", "Negative": "#DC2626", "Neutral": "#94A3B8"},
                        hole=0.45
                    )
                    fig_single.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_single, use_container_width=True)

                with st.expander("View Raw Output (JSON)"):
                    st.json([{"feature": a, "sentiment": s, "confidence": c} for a, s, c in results])

elif navigation == "Batch Dataset Analysis":
    st.markdown('<div class="main-header">Batch Review Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Analyze an entire set of reviews to discover the most talked-about features and overall satisfaction.</div>',
        unsafe_allow_html=True
    )

    data_source = st.radio(
        "Choose Data Source:",
        options=["Pre-loaded Amazon Reviews Dataset (Sample)", "Upload CSV or Excel File"],
        horizontal=True
    )

    df_raw = None
    sample_file_path = os.path.join("data", "sample_amazon_reviews.csv")

    if data_source == "Pre-loaded Amazon Reviews Dataset (Sample)":
        if os.path.exists(sample_file_path):
            df_raw = pd.read_csv(sample_file_path)
            st.success(f"Loaded Amazon reviews dataset ({len(df_raw)} reviews available).")
        else:
            st.error("Sample dataset file not found. Run clean_data.py to create it.")
    else:
        uploaded_file = st.file_uploader("Upload review file:", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".xlsx"):
                    df_raw = pd.read_excel(uploaded_file)
                else:
                    df_raw = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Loaded {len(df_raw)} reviews.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if df_raw is not None and not df_raw.empty:
        columns = df_raw.columns.tolist()
        default_col_idx = 0
        for candidate in ["body", "review", "text", "Review", "review_text", "comment", "feedback"]:
            if candidate in columns:
                default_col_idx = columns.index(candidate)
                break

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            text_column = st.selectbox("Select Review Text Column:", options=columns, index=default_col_idx)
        with col_cfg2:
            max_rows = st.slider(
                "Number of Reviews to Analyze:",
                min_value=5,
                max_value=min(200, len(df_raw)),
                value=min(25, len(df_raw)),
                step=5,
                help="Choose 20-30 reviews for quick live demonstrations."
            )

        with st.expander("Preview Selected Reviews"):
            st.dataframe(df_raw[[text_column]].head(8), use_container_width=True)

        if st.button("Run Batch Analysis", type="primary"):
            df_subset = df_raw.dropna(subset=[text_column]).head(max_rows).copy()
            total_reviews = len(df_subset)
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            extracted_aspect_records = []

            for i, (_, row) in enumerate(df_subset.iterrows()):
                review_text = str(row[text_column])
                status_text.text(f"Analyzing review {i + 1} of {total_reviews}...")
                progress_bar.progress((i + 1) / total_reviews)

                results = extract_aspect_sentiments(review_text, extractor=extractor)
                for aspect, sentiment, conf in results:
                    extracted_aspect_records.append({
                        "Review_Number": i + 1,
                        "Feature": aspect.lower(),
                        "Sentiment": sentiment,
                        "Confidence": conf,
                        "Review_Snippet": review_text[:120] + "..." if len(review_text) > 120 else review_text
                    })

            status_text.success(f"Batch analysis complete! Processed {total_reviews} reviews.")
            progress_bar.empty()

            if not extracted_aspect_records:
                st.warning("No features were detected in the sampled reviews.")
            else:
                aspect_df = pd.DataFrame(extracted_aspect_records)

                total_mentions = len(aspect_df)
                unique_aspects = aspect_df["Feature"].nunique()
                pos_rate = (aspect_df["Sentiment"] == "Positive").mean() * 100
                neg_rate = (aspect_df["Sentiment"] == "Negative").mean() * 100
                neu_rate = (aspect_df["Sentiment"] == "Neutral").mean() * 100

                st.subheader("Batch Results Overview")
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                with kpi1:
                    st.metric("Reviews Analyzed", total_reviews)
                with kpi2:
                    st.metric("Total Feature Mentions", total_mentions)
                with kpi3:
                    st.metric("Unique Features", unique_aspects)
                with kpi4:
                    st.metric("Positive Ratio", f"{pos_rate:.1f}%")
                with kpi5:
                    st.metric("Negative Ratio", f"{neg_rate:.1f}%")

                st.write("---")

                top_10_aspects_series = aspect_df["Feature"].value_counts().head(10)
                top_10_names = top_10_aspects_series.index.tolist()

                c_chart1, c_chart2 = st.columns(2)

                with c_chart1:
                    st.markdown("#### Top 10 Most Mentioned Features")
                    df_top10 = top_10_aspects_series.reset_index()
                    df_top10.columns = ["Feature", "Mention Count"]

                    fig_bar = px.bar(
                        df_top10,
                        x="Mention Count",
                        y="Feature",
                        orientation="h",
                        color="Mention Count",
                        color_continuous_scale="Blues",
                        title="Most Frequent Features"
                    )
                    fig_bar.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with c_chart2:
                    st.markdown("#### Sentiment Breakdown for Top Features")
                    df_top_filtered = aspect_df[aspect_df["Feature"].isin(top_10_names)]
                    
                    distribution = (
                        df_top_filtered.groupby(["Feature", "Sentiment"])
                        .size()
                        .reset_index(name="Count")
                    )

                    fig_stacked = px.bar(
                        distribution,
                        x="Feature",
                        y="Count",
                        color="Sentiment",
                        barmode="stack",
                        title="Positive / Negative / Neutral by Feature",
                        color_discrete_map={
                            "Positive": "#16A34A",
                            "Negative": "#DC2626",
                            "Neutral": "#94A3B8"
                        }
                    )
                    fig_stacked.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_tickangle=-30
                    )
                    st.plotly_chart(fig_stacked, use_container_width=True)

                st.write("---")
                st.subheader("Feature Frequency and Sentiment Summary Table")
                
                pivot_summary = aspect_df.groupby("Feature")["Sentiment"].value_counts().unstack(fill_value=0)
                for sent_col in ["Positive", "Negative", "Neutral"]:
                    if sent_col not in pivot_summary.columns:
                        pivot_summary[sent_col] = 0

                pivot_summary["Total Mentions"] = pivot_summary.sum(axis=1)
                sentiment_score = (pivot_summary["Positive"] - pivot_summary["Negative"]) / pivot_summary["Total Mentions"]
                pivot_summary["Net Sentiment Score"] = sentiment_score.round(2)
                pivot_summary = pivot_summary.sort_values(by="Total Mentions", ascending=False).reset_index()

                st.dataframe(pivot_summary, use_container_width=True)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_aspects = aspect_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Extracted Features (CSV)",
                        data=csv_aspects,
                        file_name="extracted_features_detailed.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    csv_summary = pivot_summary.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Feature Summary Table (CSV)",
                        data=csv_summary,
                        file_name="feature_summary_table.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
