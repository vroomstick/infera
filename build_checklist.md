# 🧠 INFERA — Full Build Checklist

This document tracks the complete build process of the Infera project, organized chronologically by file, folder, and task. Use this to manage progress and show clear project structure to reviewers, collaborators, or future employers.

---

## ✅ PHASE 0 — ENV SETUP (Completed)
- [x] Create GitHub repo
- [x] Scaffold folder and file structure
- [x] Initialize `git` and push scaffold
- [x] Create and activate `venv`
- [x] Set Python interpreter in VS Code
- [x] Create `.gitignore` and exclude `venv/`, `__pycache__/`, `data/`, `reports/`, etc.
- [x] Keep `infera_config.py` in version control (no longer hidden)
- [x] Add SEC headers config for scraping

---

## 🟨 PHASE 1 — INGEST

**`infera_config.py`**
- [x] Store `SEC_HEADERS` dictionary with proper User-Agent (SEC-compliant)
- [ ] (Optional) Add ticker → CIK lookup cache

**`ingest/sec_fetcher.py`**
- [x] Fetch CIK from SEC's public JSON mapping
- [x] Fetch latest 10-K metadata from EDGAR submissions
- [x] Construct URL to raw HTML (using accession number + primary document)
- [x] Download HTML with browser headers
- [x] Save raw HTML into `data/{ticker}_10k.html`

---

## 🟨 PHASE 2 — CLEANING AND SECTIONING

**`analyze/cleaner.py`**
- [ ] Parse HTML with `BeautifulSoup`
- [ ] Strip script/style/boilerplate/tables
- [ ] Normalize and clean up text
- [ ] Output cleaned text as `data/{ticker}_cleaned.html` or string

**`analyze/segmenter.py`**
- [ ] Regex-based detection of "Item 1A", "Item 7", etc.
- [ ] Extract Risk Factors / MD&A / Business sections
- [ ] Return section text as dictionary object
- [ ] (Optional) Save to segmented text file for debugging

---

## 🟨 PHASE 3 — RELEVANCE SCORING (ML)

**`analyze/scorer.py`**
- [ ] Use `TfidfVectorizer` and `cosine_similarity`
- [ ] Compare section/paragraph vectors to prompt text (e.g. "emerging risks")
- [ ] Return top-N chunks ranked by relevance

---

## 🟨 PHASE 4 — SUMMARIZATION WITH GPT-4o

**`analyze/summarizer.py`**
- [ ] Feed relevant chunks into GPT-4o via `openai` API
- [ ] Use prompt templates to extract summaries or red flags
- [ ] Return structured dictionary of summarized sections

---

## 🟨 PHASE 5 — OUTPUT GENERATION

**`output/formatter.py`**
- [ ] Load summaries into a Markdown template (via Jinja2)
- [ ] Include headings, bullet points, and timestamp
- [ ] Output Markdown report string or save to file

**`output/exporter.py`**
- [ ] Convert Markdown → HTML → PDF using `weasyprint` (or `pdfkit`)
- [ ] Save final PDF to `reports/{ticker}_report.pdf`

---

## 🟨 PHASE 6 — INTERFACE (OPTIONAL)

**`interface/dashboard.py`**
- [ ] Build basic Streamlit dashboard with input field
- [ ] Run `run_pipeline(ticker)` on button click
- [ ] Display Markdown preview
- [ ] Offer PDF download link

---

