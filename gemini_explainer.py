import google.generativeai as genai
genai.configure(api_key="AIzaSyCnV4-0O93_ulRyhGI1rrlLjrgHmBl93p8")
gemini = genai.GenerativeModel("gemini-1.5-flash")
def answer_question(question: str, results: dict) -> str:
    explanation = explain_bias(results)
    return f"{explanation}\n\nQuestion: {question}"
def explain_bias(results: dict) -> str:
    """
    Generates an explanation of bias based on the provided results.
    This is a placeholder implementation.
    """
    bias_score = results.get("bias_score", "N/A")
    sensitive_col = results.get("sensitive_col", "N/A")
    groups = ", ".join(results.get("groups", []))
    is_bias = results.get("is_bias", False)

    explanation = f"Analysis of bias detection results:\n"
    explanation += f"  - Bias detected: {is_bias}\n"
    explanation += f"  - Bias score: {bias_score}\n"
    explanation += f"  - Sensitive column: {sensitive_col}\n"
    explanation += f"  - Affected groups: {groups}"
    return explanation

if __name__ == "__main__":
    sample_results = {
        "accuracy": 87,
        "bias_score": 0.15,
        "sensitive_col": "gender",
        "groups": ["male", "female"],
        "is_bias": True
    }

    print(explain_bias(sample_results))
    def answer_question(question: str, results: dict) -> str:
        prompt = f"Bias report: {results}"
        return prompt
fake_results = {
    "accuracy": 85.2,
    "bias_score": 0.19,
    "raw_dpd": 0.19,
    "sensitive_col": "gender",
    "groups": ["Male", "Female"],
    "is_biased": True
}

explanation = explain_bias(fake_results)
print("EXPLANATION:")
print(explanation)
print()
answer = answer_question("How can we fix this bias?", fake_results)
print("CHATBOT ANSWER:")
print(answer)
def get_severity(bias_score: float) -> str:
    if bias_score < 0.05:
        return "None — model is fair"
    elif bias_score < 0.10:
        return "Low — minor differences exist"
    elif bias_score < 0.20:
        return "Medium — action recommended"
    else:
        return "High — significant bias detected"
   
GEMINI_API_KEY="AIzaSyCnV4-0O93_ulRyhGI1rrlLjrgHmBl93p8"
from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))