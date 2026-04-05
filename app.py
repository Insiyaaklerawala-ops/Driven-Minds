import streamlit as st
import pandas as pd
import plotly.express as px

from bias_engine import analyze_bias
from gemini_explainer import explain_bias, answer_question
from report_generator import generate_pdf

# ---------------------------
# ⚙️ CONFIG
# ---------------------------
st.set_page_config(
    page_title="Unbiased AI",
    page_icon="⚖️",
    layout="wide"
)

# ---------------------------
# 🎨 HEADER
# ---------------------------
st.title("⚖️ Unbiased AI — Bias Detector")
st.caption("Detect and explain hidden discrimination in AI decision systems")

# ---------------------------
# 📂 FILE UPLOAD
# ---------------------------
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# ---------------------------
# 📊 LOAD DATA
# ---------------------------
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Error reading file: {e}")
    st.stop()

if df.empty:
    st.error("❌ Uploaded file is empty.")
    st.stop()

st.success(f"Loaded {len(df):,} rows · {len(df.columns)} columns")

with st.expander("Preview data"):
    st.dataframe(df.head(10))

# ---------------------------
# 🎯 COLUMN SELECTORS
# ---------------------------
col1, col2 = st.columns(2)

label_col = col1.selectbox("Target column", df.columns)

sensitive_col = col2.selectbox(
    "Sensitive column",
    [col for col in df.columns if col != label_col]
)

# Prevent same column
if label_col == sensitive_col:
    st.error("Target and sensitive column cannot be the same.")
    st.stop()

st.divider()

# ---------------------------
# 🚀 ANALYZE BUTTON
# ---------------------------
if st.button("Analyze Bias", type="primary"):

    valid_data = df[[label_col, sensitive_col]].dropna()
    st.write(f"Valid rows after cleaning: {valid_data.shape[0]}")

    if valid_data.shape[0] < 50:
        st.warning("⚠️ Small dataset after cleaning — results may be unreliable.")

    with st.spinner("Analyzing... please wait"):
        try:
            st.session_state.results = analyze_bias(df, label_col, sensitive_col)
            st.session_state.explanation = explain_bias(st.session_state.results)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

# ---------------------------
# 📊 SHOW RESULTS
# ---------------------------
if "results" in st.session_state:

    results = st.session_state.results
    explanation = st.session_state.explanation

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)

    # Accuracy
    c1.metric("Model Accuracy", f"{results.get('accuracy', 'N/A')}%")

    # Bias score
    c2.metric(
        "Bias Score",
        results.get("bias_score", "N/A"),
        delta="Above threshold" if results.get("is_biased") else "Below threshold",
        delta_color="inverse"
    )

    # ✅ SIMPLE VERDICT (NO COLOR)
    verdict = "Bias Found" if results.get("is_biased") else "Looks Fair"

    c3.metric(
        "Verdict",
        verdict,
        delta="Needs attention" if results.get("is_biased") else "All good",
        delta_color="inverse"
    )

    st.info(f"AI Analysis: {explanation}")

    # ---------------------------
    # 📈 VISUALIZATION
    # ---------------------------
    groups = results.get("groups", [])
    bias_score = results.get("bias_score", 0)
    group_rates = results.get("group_rates", {})

    # Preferred graph
    if group_rates:
        fig = px.bar(
            x=list(group_rates.keys()),
            y=list(group_rates.values()),
            labels={
                "x": results.get("sensitive_col", ""),
                "y": "Positive prediction rate"
            },
            title=f"Prediction rate per group — {results.get('sensitive_col', '')}",
            color=list(group_rates.values()),
            color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"]
        )

        fig.update_layout(coloraxis_showscale=False)

        fig.add_hline(
            y=sum(group_rates.values()) / len(group_rates),
            line_dash="dash",
            annotation_text="Average rate"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Fallback graph
    elif groups:
        color = "#E24B4A" if results.get("is_biased") else "#1D9E75"

        fig = px.bar(
            x=groups,
            y=[bias_score] * len(groups),
            labels={
                "x": results.get("sensitive_col", ""),
                "y": "Bias score"
            },
            title=f"Bias across {results.get('sensitive_col', '')}",
            color_discrete_sequence=[color]
        )

        fig.add_hline(y=0.1, line_dash="dash", annotation_text="Bias threshold")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------
    # 📄 PDF DOWNLOAD
    # ---------------------------
    try:
        pdf_path = generate_pdf(results, explanation)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, "bias_report.pdf")
    except Exception as e:
        st.warning(f"⚠️ PDF error: {e}")

    st.divider()

    # ---------------------------
    # 💬 CHAT
    # ---------------------------
    st.subheader("Ask about this report")

    with st.form("chat_form"):
        question = st.text_input("Ask a question")
        submitted = st.form_submit_button("Get Answer")

    if submitted and question:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(question, results)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("Run analysis to see results.")

# ---------------------------
# 📌 SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("About this tool")
    st.markdown("""
This tool detects **hidden bias** in AI models.

**How it works:**
1. Upload dataset  
2. Choose prediction column  
3. Choose sensitive attribute  
4. Model is trained  
5. Bias is measured + explained  

**Bias score guide:**
- 0.00 – 0.05 → No bias  
- 0.05 – 0.10 → Minor bias  
- 0.10 – 0.20 → Significant bias  
- 0.20+ → Severe bias  
""")