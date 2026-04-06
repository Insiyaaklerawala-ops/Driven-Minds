import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# ✅ LOAD API KEY (Cloud + Local)
# ---------------------------
try:
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found (Streamlit secrets or .env)")

# ---------------------------
# ✅ GROQ CLIENT
# ---------------------------
from groq import Groq
client = Groq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"   # ✅ confirmed working

FALLBACK = (
    "This model shows significant bias — one group is "
    "considerably less likely to receive a positive outcome "
    "compared to another group with similar profiles. "
    "This must be addressed before real-world deployment."
)

# ---------------------------
# ✅ CORE CALL FUNCTION
# ---------------------------
def _call(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return FALLBACK + f" (Error: {str(e)[:80]})"


# ---------------------------
# ✅ EXPLAIN BIAS
# ---------------------------
def explain_bias(results: dict) -> str:
    prompt = f"""Explain these AI bias results in 3 simple sentences
to a non-technical person. No jargon.

Bias score: {results['bias_score']} (above 0.1 = significant bias)
Column checked: {results['sensitive_col']}
Groups found: {results['groups']}
Bias detected: {results['is_biased']}
Model accuracy: {results['accuracy']}%

Explain what the model predicts, what the bias means for
real people, and one way to fix it."""
    return _call(prompt)


# ---------------------------
# ✅ Q&A FUNCTION
# ---------------------------
def answer_question(question: str, results: dict) -> str:
    prompt = f"""Bias report: {results}
User question: {question}
Answer in 2 simple sentences. No technical terms."""
    return _call(prompt)


# ---------------------------
# ✅ MITIGATION EXPLAINER
# ---------------------------
def explain_mitigation(before: dict, after: dict) -> str:
    improvement = round(
        before['bias_score'] - after['after_bias_score'], 3
    )

    pct_improvement = round(
        (improvement / before['bias_score']) * 100
    ) if before['bias_score'] > 0 else 0

    prompt = f"""
You are an AI fairness expert. Explain what happened when we
applied bias mitigation to this model. Use simple language only.

Before mitigation:
- Bias score: {before['bias_score']} (significant bias)
- Accuracy: {before['accuracy']}%

After mitigation:
- Bias score: {after['after_bias_score']} (much lower)
- Accuracy: {after['after_accuracy']}%
- Improvement: {pct_improvement}% reduction in bias

Explain in 3-4 sentences:
1. What the mitigation did in simple terms
2. How much the bias improved
3. Whether there was a trade-off in accuracy
4. Whether the model is now safe to use

Use plain language. No technical jargon.
No mentions of algorithm names.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not generate explanation: {str(e)}"