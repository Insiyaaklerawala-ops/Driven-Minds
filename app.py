import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bias_engine import analyze_bias, mitigate_bias
from gemini_explainer import explain_bias, answer_question, explain_mitigation
from report_generator import generate_pdf

# ---------------------------
# ⚡ CACHE FUNCTIONS (MAJOR SPEED BOOST)
# ---------------------------
@st.cache_data
def run_analysis(df, label_col, sensitive_col):
    results = analyze_bias(df, label_col, sensitive_col)
    explanation = explain_bias(results)
    return results, explanation

@st.cache_data
def run_mitigation(df, label_col, sensitive_col):
    return mitigate_bias(df, label_col, sensitive_col)

@st.cache_data
def get_mitigation_explanation(before, after):
    return explain_mitigation(before, after)

# ---------------------------
# ✅ SESSION STATE INIT
# ---------------------------
for key in ["results", "explanation", "after", "mitigation_explanation"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
            results, explanation = run_analysis(df, label_col, sensitive_col)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    st.session_state.results = results
    st.session_state.explanation = explanation
    st.session_state.after = None  # reset mitigation

# ---------------------------
# 📊 SHOW RESULTS
# ---------------------------
if st.session_state.results is not None:

    results = st.session_state.results
    explanation = st.session_state.explanation

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Model Accuracy", f"{results.get('accuracy', 'N/A')}%")

    c2.metric(
        "Bias Score",
        results.get("bias_score", "N/A"),
        delta="Above threshold" if results.get("is_biased") else "Below threshold",
        delta_color="inverse"
    )

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
    group_rates = results.get("group_rates", {})

    if group_rates:
        fig = px.bar(
            x=list(group_rates.keys()),
            y=list(group_rates.values()),
            title="Prediction rate per group",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------
    # 📄 PDF
    # ---------------------------
    pdf_path = generate_pdf(
        st.session_state.results,
        st.session_state.explanation,
        after=st.session_state.get('after', None),
        mit_explanation=st.session_state.get('mit_explanation', None)
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Full PDF Report",
            data=f,
            file_name="bias_report.pdf",
            mime="application/pdf"
        )

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
            answer = answer_question(question, results)
            st.success(answer)

# ---------------------------
# ⚡ FIX BIAS SECTION (OPTIMIZED)
# ---------------------------
if st.session_state.results and st.session_state.results["is_biased"]:

    st.divider()
    st.subheader("Fix the Bias")
    st.caption("Click below to apply mitigation and see improvement")

    if st.button("Fix Bias Now", type="primary"):

        st.info("⚙️ Running mitigation...")

        with st.spinner("Applying bias mitigation..."):
            after = run_mitigation(df, label_col, sensitive_col)

        st.success("✅ Mitigation complete!")

        with st.spinner("Generating explanation..."):
            mit_explanation = get_mitigation_explanation(
                st.session_state.results, after
            )

        st.session_state.after = after
        st.session_state.mitigation_explanation = mit_explanation

# ---------------------------
# 📊 SHOW MITIGATION RESULTS
# ---------------------------
if st.session_state.after:

    before_score = st.session_state.results["bias_score"]
    after_score = st.session_state.after["after_bias_score"]

    st.subheader("Before vs After Mitigation")

    col1, col2 = st.columns(2)

    col1.metric("Before", before_score)
    col2.metric("After", after_score)

    improvement = round(before_score - after_score, 3)
    st.metric("Bias reduction", improvement)

    st.info(st.session_state.mitigation_explanation)

    fig = go.Figure(data=[
        go.Bar(name='Before', x=['Bias'], y=[before_score]),
        go.Bar(name='After', x=['Bias'], y=[after_score])
    ])
    st.plotly_chart(fig, use_container_width=True)

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