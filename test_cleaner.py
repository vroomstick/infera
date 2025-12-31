from analyze.cleaner import clean_html

with open("data/AAPL_10K.html", "r", encoding="utf-8") as f:
    raw_html = f.read()

cleaned = clean_html(raw_html)
print(cleaned[:1000])  # Preview what the cleaner returns
