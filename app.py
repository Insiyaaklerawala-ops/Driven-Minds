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

st.title("⚖️ Unbiased AI — Bias Detector")
st.caption("Upload any dataset to find hidden discrimination in AI models")

# ---------------------------
# 📂 FILE UPLOAD
# ---------------------------
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started. Try the Adult Income dataset.")
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

with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10))

# ---------------------------
# 🎯 COLUMN SELECTORS
# ---------------------------
col1, col2 = st.columns(2)

label_col = col1.selectbox(
    "What are we predicting?",
    df.columns
)

sensitive_col = col2.selectbox(
    "Which column might cause bias?",
    [col for col in df.columns if col != label_col]
)

st.divider()

# ---------------------------
# 🚀 ANALYZE BUTTON
# ---------------------------
if st.button("Analyze Bias", type="primary"):

    valid_data = df[[label_col, sensitive_col]].dropna()

    if valid_data.shape[0] < 10:
        st.error("❌ Not enough valid data after cleaning.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            st.session_state.results = analyze_bias(df, label_col, sensitive_col)
            st.session_state.explanation = explain_bias(st.session_state.results)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# ---------------------------
# 📊 SHOW RESULTS (PERSISTENT)
# ---------------------------
if "results" in st.session_state:

    results = st.session_state.results
    explanation = st.session_state.explanation

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{results['accuracy']}%")
    c2.metric("Bias Score", results['bias_score'])

    verdict = "Bias Found" if results['is_biased'] else "Looks Fair"
    c3.metric("Verdict", verdict)

    st.info(f"AI Analysis: {explanation}")

    # ---------------------------
    # 📈 VISUALIZATION
    # ---------------------------
    color = "#E24B4A" if results['is_biased'] else "#1D9E75"

    fig = px.bar(
        x=results['groups'],
        y=[results['bias_score']] * len(results['groups']),
        labels={"x": results['sensitive_col'], "y": "Bias score"},
        title=f"Bias score across {results['sensitive_col']} groups",
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
            st.download_button(
                "Download PDF Report",
                f,
                "bias_report.pdf"
            )
    except Exception as e:
        st.warning(f"⚠️ PDF error: {e}")

    st.divider()

    # ---------------------------
    # 💬 CHAT (FIXED)
    # ---------------------------
    st.subheader("Ask about this report")

    with st.form("chat_form"):
        question = st.text_input(
            "Type your question...",
            placeholder="Why is this biased? How do we fix it?"
        )
        submitted = st.form_submit_button("Get Answer")

    if submitted and question:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(question, results)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

elif "results" not in st.session_state:
    st.info("Run analysis to see results and ask questions.")