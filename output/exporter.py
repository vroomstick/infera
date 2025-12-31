# output/exporter.py

import os
import markdown2
import pdfkit

def convert_markdown_to_pdf(md_path, pdf_path):
    """
    Converts a Markdown file to PDF using pdfkit.

    Args:
        md_path (str): Path to the Markdown file
        pdf_path (str): Desired output path for PDF
    """
    try:
        # Read Markdown and convert to HTML
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        html_content = markdown2.markdown(md_content)

        # Save temporary HTML file
        tmp_html = md_path.replace(".md", ".html")
        with open(tmp_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Convert HTML to PDF using pdfkit
        pdfkit.from_file(tmp_html, pdf_path)
        print("🧾 PDF created with pdfkit.")

    except Exception as e:
        print(f"❌ PDF export failed: {e}")

    finally:
        # Clean up temporary HTML file
        if os.path.exists(tmp_html):
            os.remove(tmp_html)
