import streamlit as st
import pandas as pd
import plotly.express as px

from bias_engine import analyze_bias
from gemini_explainer import explain_bias, answer_question
from report_generator import generate_pdf

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
# 📊 LOAD DATA SAFELY
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
    df.columns,
    help="The outcome column e.g. 'income', 'hired'"
)

sensitive_col = col2.selectbox(
    "Which column might cause bias?",
    [col for col in df.columns if col != label_col],
    help="e.g. 'gender', 'race', 'age_group'"
)

st.divider()

# ---------------------------
# 🚀 ANALYZE BUTTON
# ---------------------------
if st.button("Analyze Bias", type="primary"):

    # ✅ Prevent empty dataset AFTER cleaning
    valid_data = df[[label_col, sensitive_col]].dropna()

    if valid_data.shape[0] < 10:
        st.error("❌ Not enough valid data after removing missing values. Try different columns.")
        st.stop()

    with st.spinner("Analyzing your dataset... please wait"):
        try:
            results = analyze_bias(df, label_col, sensitive_col)
            explanation = explain_bias(results)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.stop()

    # ---------------------------
    # 📊 RESULTS
    # ---------------------------
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{results['accuracy']}%")
    c2.metric(
        "Bias Score",
        results['bias_score'],
        help="0 = fair · above 0.1 = significant bias"
    )

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

    fig.add_hline(
        y=0.1,
        line_dash="dash",
        annotation_text="Bias threshold (0.10)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------
    # 📄 PDF DOWNLOAD
    # ---------------------------
    try:
        pdf_path = generate_pdf(results, explanation)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="bias_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.warning(f"⚠️ Could not generate PDF: {e}")

    st.divider()

    # ---------------------------
    # 💬 Q&A SECTION
    # ---------------------------
    st.subheader("Ask about this report")

    question = st.text_input(
        "Type your question...",
        placeholder="Why is this biased? How do we fix it?"
    )

    if question:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(question, results)
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")