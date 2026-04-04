from fpdf import FPDF
import datetime

# ✅ Function to clean unsupported Unicode characters
def clean_text(text):
    return str(text).replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"')

def generate_pdf(results: dict, explanation: str) -> str:

    pdf = FPDF()
    pdf.add_page()

    # TITLE
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Unbiased AI - Bias Detection Report", ln=True)

    # DATE in small grey text
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 8, f"Generated on: {datetime.date.today()}", ln=True)
    pdf.ln(4)

    # HORIZONTAL LINE (divider)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Reset text color back to black
    pdf.set_text_color(0, 0, 0)

    # SECTION HEADING
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Bias Metrics", ln=True)

    # DATA ROWS
    rows = [
        ("Model accuracy",   f"{results['accuracy']}%"),
        ("Bias score",       str(results['bias_score'])),
        ("Column analyzed",  results['sensitive_col']),
        ("Groups found",     ", ".join(str(g) for g in results['groups'])),
        ("Verdict",          "BIASED" if results['is_biased'] else "FAIR"),
    ]

    for label, value in rows:
        value = clean_text(value)  # ✅ Clean text

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(65, 8, label + ":", ln=False)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, value, ln=True)

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # AI EXPLANATION SECTION
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "AI Explanation", ln=True)

    pdf.set_font("Helvetica", "", 11)
    explanation = clean_text(explanation)  # ✅ Clean explanation
    pdf.multi_cell(0, 7, explanation)

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # RECOMMENDATIONS
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Recommended Actions", ln=True)

    pdf.set_font("Helvetica", "", 11)

    if results['is_biased']:
        actions = [
            "1. Review and rebalance the training dataset",
            "2. Apply Fairlearn reweighing to reduce bias",
            "3. Re-evaluate the model before deployment",
            "4. Set up ongoing fairness monitoring",
        ]
    else:
        actions = [
            "1. Model appears fair - continue monitoring",
            "2. Re-test periodically as new data comes in",
        ]

    for action in actions:
        action = clean_text(action)  # ✅ Clean actions
        pdf.cell(0, 8, action, ln=True)

    # SAVE PDF
    path = "bias_report.pdf"
    pdf.output(path)

    return path


# TEST RUN
if __name__ == "__main__":

    fake_results = {
        "accuracy": 85.2,
        "bias_score": 0.19,
        "sensitive_col": "gender",
        "groups": ["Male", "Female"],
        "is_biased": True
    }

    fake_explanation = (
        "This model predicts whether someone earns over $50K per year. "
        "We found significant gender bias — the model is 19% less likely "
        "to predict high income for women with the same qualifications "
        "as men. To fix this, the training data should be rebalanced "
        "so both genders are equally represented in high-income examples."
    )

    path = generate_pdf(fake_results, fake_explanation)
    print(f"PDF created at: {path}")
    print("Open the file to check it looks correct.")