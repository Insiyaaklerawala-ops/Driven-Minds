import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Unbiased AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Unbiased AI — Bias Detector")
st.caption("Upload any dataset to find hidden discrimination in AI models")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started. Try the Adult Income dataset.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {len(df):,} rows · {len(df.columns)} columns")

with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10))

col1, col2 = st.columns(2)

label_col = col1.selectbox(
    "What are we predicting?",
    df.columns,
    help="The outcome column e.g. 'income', 'hired', 'approved'"
)

sensitive_col = col2.selectbox(
    "Which column might cause bias?",
    df.columns,
    help="e.g. 'gender', 'race', 'age_group'"
)

st.divider()

if st.button("Analyze Bias", type="primary"):
    st.info("Analysis will appear here. Logic coming in Hour 2!")

    FAKE_RESULTS = {
        "accuracy": 85.2,
        "bias_score": 0.19,
        "sensitive_col": "gender",
        "groups": ["Male", "Female"],
        "is_biased": True
    }

    FAKE_EXPLANATION = (
        "This model predicts whether someone earns over $50K per year. "
        "We found significant gender bias — the model is 19% less likely "
        "to predict high income for women compared to men with similar "
        "qualifications. To fix this, the training data should be rebalanced "
        "so both genders are equally represented in high-income examples."
    )

    results = FAKE_RESULTS
    explanation = FAKE_EXPLANATION

    c1, c2, c3 = st.columns(3)

    c1.metric("Model Accuracy", f"{results['accuracy']}%")

    c2.metric(
        "Bias Score",
        results['bias_score'],
        help="0 = fair · above 0.1 = significant bias"
    )

    verdict = "Bias Found" if results['is_biased'] else "Looks Fair"
    c3.metric("Verdict", verdict)

    st.divider()

    st.subheader("🧠 Explanation")
    st.write(explanation)
    import plotly.express as px

    st.info(f"AI Analysis: {explanation}")

    color = "#E24B4A" if results['is_biased'] else "#1D9E75"
    fig = px.bar(
        x=results['groups'],
        y=[results['bias_score']] * len(results['groups']),
        labels={"x": results['sensitive_col'], "y": "Bias score"},
        title=f"Bias across {results['sensitive_col']} groups",
        color_discrete_sequence=[color]
    )
    fig.add_hline(y=0.1, line_dash="dash",
                  annotation_text="Bias threshold (0.10)")
    st.plotly_chart(fig, width='stretch')
    st.divider()
    st.subheader("Ask about this report")
    question = st.text_input(
        "Type your question...",
        placeholder="Why is this biased? How do we fix it?"
    )
    if question:
        st.write("Thinking...")
        st.write(
            "This is a placeholder answer. On Day 3 this will "
            "call the real Gemini chatbot from M2's file."
        )
    with st.sidebar:
     st.header("Dataset Info")
    if uploaded:
        st.metric("Total rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        st.write("Columns found:")
        st.write(df.columns.tolist())
        