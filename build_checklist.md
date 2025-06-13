from pathlib import Path

# Create the content for build_checklist.md
checklist_content = """# 🧠 INFERA — Full Build Checklist

This document tracks the complete build process of the Infera project, organized chronologically by file, folder, and task. Use this to manage progress and show clear project structure to reviewers, collaborators, or future employers.

---

## ✅ PHASE 0 — ENV SETUP (Completed)
- [x] Create GitHub repo
- [x] Scaffold folder and file structure
- [x] Initialize `git` and push scaffold
- [x] Create and activate `venv`
- [x] Set Python interpreter in VS Code
- [x] Create `.gitignore` and exclude `venv/`, `infera_config.py`, etc.

---

## 🟨 PHASE 1 — INGEST

**`infera_config.py`**
- [ ] Store your `sec-api.io` API key securely as a constant

**`ingest/sec_fetcher.py`**
- [ ] Import API key from `infera_config.py`
- [ ] Use `requests` to query and fetch a full 10-K filing
- [ ] Save raw HTML into `data/raw/`

---

## 🟨 PHASE 2 — CLEANING AND SECTIONING

**`analyze/cleaner.py`**
- [ ] Clean raw HTML using `BeautifulSoup`
- [ ] Remove boilerplate, whitespace, tables, etc.
- [ ] Output cleaned text into `data/clean/`

**`analyze/segmenter.py`**
- [ ] Regex-based detection of Risk Factors / MD&A
- [ ] Split text into dictionary of sections
- [ ] Save segmented sections as JSON or text blobs

---

## 🟨 PHASE 3 — RELEVANCE SCORING (ML)

**`analyze/scorer.py`**
- [ ] Run TF-IDF vectorization over Risk section
- [ ] Score relevance of each paragraph/sentence
- [ ] Rank based on similarity to risk keywords or prior filings
- [ ] Output top-N highest scoring chunks

---

## 🟨 PHASE 4 — SUMMARIZATION WITH GPT-4o

**`analyze/summarizer.py`**
- [ ] Use OpenAI’s GPT-4o API
- [ ] Feed it top-ranked risk chunks
- [ ] Prompt it to extract red flags, summarize trends
- [ ] Return structured plain-English output

---

## 🟨 PHASE 5 — OUTPUT GENERATION

**`output/formatter.py`**
- [ ] Format GPT output into structured Markdown
- [ ] Add headers, timestamps, and context info

**`output/exporter.py`**
- [ ] Convert Markdown → PDF
- [ ] Save final report into `/reports/`

---

## 🟨 PHASE 6 — INTERFACE (OPTIONAL)

**`interface/dashboard.py`**
- [ ] Build basic Streamlit dashboard
- [ ] Allow user to enter company name or CIK
- [ ] Display results, summaries, and downloadable PDF

---

## 🧠 FINAL PRODUCT: INFERA

A modular, production-grade AI tool that:
- Pulls and cleans SEC 10-Ks
- Extracts and ranks key narrative sections
- Summarizes them via ML + GPT
- Outputs reports in human-readable form
- Offers an optional UI for usability
"""

# Write it to the appropriate file path
file_path = Path("build_checklist.md")
file_path.write_text(checklist_content)

file_path.name  # Return the filename to show what was created
