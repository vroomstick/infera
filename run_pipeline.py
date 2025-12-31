# run_pipeline.py

import os
import argparse
from datetime import datetime

from analyze.cleaner import clean_html
from analyze.segmenter import get_risk_section
from analyze.scorer import score_sections
from analyze.summarizer import summarize_section
from output.formatter import format_to_markdown
# from output.exporter import convert_markdown_to_pdf  # PDF export deferred to V2

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "risk_factors.txt")
SECTION_LABEL = "Risk Factors"

# === Helper ===
def preview(text, label):
    print(f"\n📌 {label} Preview:")
    safe = text.strip().replace("\n", " ")[:300]
    print(safe + (" ..." if len(text) > 300 else ""))


def run_pipeline(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    print(f"\n🚀 Running Infera on file: {file_path}")

    # Step 1: Clean HTML
    cleaned_text = clean_html(file_path)
    preview(cleaned_text, "Cleaned HTML")

    # Step 2: Extract Risk Factors
    section_text = get_risk_section(cleaned_text)
    if not section_text:
        raise ValueError("❌ Risk Factors section not found.")
    preview(section_text, "Extracted Risk Factors")

    # Step 3: Score Paragraphs
    top_chunks = score_sections(section_text, top_n=5)
    top_text = "\n\n".join(top_chunks)
    preview(top_text, "Top Scored Paragraphs")

    # Step 4: Summarize with GPT
    summary = summarize_section(
        section_name=SECTION_LABEL,
        section_text=top_text,
        prompt_path=PROMPT_PATH
    )
    preview(summary, "GPT Summary")

    # Step 5: Format Markdown
    ticker = os.path.basename(file_path).split("_")[0]
    today = datetime.today().strftime("%Y-%m-%d")
    markdown = format_to_markdown(
        ticker=ticker,
        section_summaries={SECTION_LABEL: summary},
        date=today
    )

    # Step 6: Save Output
    os.makedirs("reports", exist_ok=True)
    md_path = f"reports/{ticker}_{today}.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"\n✅ Markdown saved: {md_path}")

    # === PDF Export (Deferred to V2) ===
    # pdf_path = f"reports/{ticker}_{today}.pdf"
    # convert_markdown_to_pdf(md_path, pdf_path)
    # print(f"📄 PDF exported: {pdf_path}")
    print("📄 PDF export skipped in V1. Markdown output only.")

    # === for V2 ===
    # - Re-enable the PDF export block above once a stable backend is chosen
    # - Preferred stack: headless Chrome + markdown-to-HTML render + Puppeteer or Playwright
    # - Optional: Support multiple formats (PDF, HTML, DOCX) via CLI flag
    # - Consider `pdfkit` fallback if installation hurdles are resolved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Infera pipeline on a local 10-K HTML file.")
    parser.add_argument("--file", required=True, help="Path to the 10-K HTML file (e.g., data/AAPL_10K.html)")
    args = parser.parse_args()

    run_pipeline(args.file)
