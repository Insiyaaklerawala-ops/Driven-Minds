import pandas as pd
from app.core.bias_engine import analyze_bias
from app.core.explainer import explain_bias
from app.core.report_generator import generate_pdf

df = pd.read_csv("adult.csv")

print("Step 1: Running bias detection...")
results = analyze_bias(df, "income", "gender")
print("Bias score:", results["bias_score"])

print("\nStep 2: Generating AI explanation...")
explanation = explain_bias(results)
print(explanation)

print("\nStep 3: Generating PDF report...")

path = generate_pdf(results, explanation)
print("PDF saved to:", path)

print("\nFull chain working! Ready for Day 3.")