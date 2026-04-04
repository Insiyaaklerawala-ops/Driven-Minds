import os
from dotenv import load_dotenv
from google import genai

# -------------------- LOAD ENV --------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# -------------------- CREATE CLIENT --------------------
client = genai.Client(api_key=api_key)


# -------------------- FUNCTION 1: EXPLAIN BIAS --------------------
def explain_bias(results):
    prompt = f"""
    You are an AI fairness expert.

    Analyze the following model results and explain clearly if there is bias.

    Results:
    {results}

    Explain in simple terms:
    - What the bias means
    - Why it is a problem
    - Which group is affected
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        return f"Error generating explanation: {e}"


# -------------------- FUNCTION 2: ANSWER QUESTION --------------------
def answer_question(question, results):
    prompt = f"""
    You are an AI fairness expert helping improve machine learning models.

    Model Results:
    {results}

    User Question:
    {question}

    Give a clear, practical answer with actionable steps.
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        return f"Error generating answer: {e}"