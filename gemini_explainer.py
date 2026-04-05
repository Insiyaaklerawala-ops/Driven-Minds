import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

# -------------------- LOAD ENV --------------------
load_dotenv()

# Get API key (Streamlit Cloud OR local .env)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

# Safety check
if not api_key:
    st.error("❌ API key not found. Add it to .env or secrets.toml")
    st.stop()

# -------------------- GEMINI SETUP --------------------
genai.configure(api_key=api_key)

# ✅ FIXED MODEL NAME
gemini = genai.GenerativeModel("gemini-1.5-flash-latest")


# -------------------- FUNCTIONS --------------------

def explain_bias(results: dict) -> str:
    """Generate a simple bias explanation."""
    bias_score = results.get("bias_score", "N/A")
    sensitive_col = results.get("sensitive_col", "N/A")
    groups = ", ".join(results.get("groups", []))
    is_biased = results.get("is_biased", False)

    explanation = f"""
Bias Analysis Report:
- Bias detected: {is_biased}
- Bias score: {bias_score}
- Sensitive column: {sensitive_col}
- Affected groups: {groups}
"""
    return explanation.strip()


def answer_question(question: str, results: dict) -> str:
    """Use Gemini to answer user questions about bias."""
    explanation = explain_bias(results)

    prompt = f"""
You are an AI fairness expert.

Here is the bias report:
{explanation}

User question:
{question}

Give clear, practical, and actionable solutions.
"""

    try:
        response = gemini.generate_content(prompt)

        # ✅ safer response handling
        if hasattr(response, "text"):
            return response.text
        else:
            return "⚠️ No response generated."

    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_severity(bias_score: float) -> str:
    """Categorize bias severity."""
    if bias_score < 0.05:
        return "None — model is fair"
    elif bias_score < 0.10:
        return "Low — minor differences exist"
    elif bias_score < 0.20:
        return "Medium — action recommended"
    else:
        return "High — significant bias detected"


