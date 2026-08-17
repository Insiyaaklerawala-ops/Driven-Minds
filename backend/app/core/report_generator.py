import os
import uuid
import datetime
from fpdf import FPDF

REPORTS_DIR = "generated_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def clean_text(text) -> str:
    return (
        str(text)
        .replace("—", "-")
        .replace("–", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def generate_pdf(results: dict, explanation: str,
                  after: dict = None,
                  mit_explanation: str = None) -> str:
    pdf = FPDF()
    pdf.add_page()

    # TITLE
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Unbiased AI - Bias Detection Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 8, f"Generated on: {datetime.date.today()}", ln=True)
    pdf.ln(4)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    # BIAS METRICS
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Bias Metrics", ln=True)

    rows = [
        ("Model accuracy",  f"{results['accuracy']}%"),
        ("Bias score",      str(results['bias_score'])),
        ("Column analyzed", results['sensitive_col']),
        ("Groups found",    ", ".join(str(g) for g in results['groups'])),
        ("Verdict",         "BIASED" if results['is_biased'] else "FAIR"),
    ]

    for label, value in rows:
        value = clean_text(value)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(65, 8, label + ":", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, value, ln=True)

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # AI EXPLANATION
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "AI Explanation", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, clean_text(explanation))

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
        pdf.cell(0, 8, clean_text(action), ln=True)

    # MITIGATION SECTION
    if after is not None:
        pdf.ln(4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Bias Mitigation Results", ln=True)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(65, 8, "Before bias score:", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(results['bias_score']), ln=True)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(65, 8, "After bias score:", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(after['after_bias_score']), ln=True)

        improvement = round(results['bias_score'] - after['after_bias_score'], 3)
        pct = round((improvement / results['bias_score']) * 100) if results['bias_score'] > 0 else 0

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(65, 8, "Improvement:", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{pct}% reduction in bias", ln=True)

        if mit_explanation:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "What was done:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, clean_text(mit_explanation))

    # UNIQUE FILENAME per request — avoids overwrites under concurrent users
    filename = f"bias_report_{uuid.uuid4().hex}.pdf"
    path = os.path.join(REPORTS_DIR, filename)
    pdf.output(path)

    return path