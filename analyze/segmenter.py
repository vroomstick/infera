# analyze/segmenter.py

import re

def extract_between_items(text):
    pattern = re.compile(r"item\s*1a[\s:–.-]*risk\s*factors(.*?)(?=item\s*1b)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        section = match.group(1).strip()
        if len(section.split()) > 100:
            print("✅ Primary extraction (Item 1A → 1B) succeeded.")
            print("🪵 Preview:", repr(section[:500]))
            return section
        else:
            print(f"⚠️ Primary match too short (length: {len(section.split())} words). Fallback triggered.")
    else:
        print("❌ Could not locate exact Item 1A → 1B section.")
    return None

def fallback_collect_risk_paragraphs(text):
    print("⚠️ Primary failed. Fallback: collecting risk-related paragraphs.")
    paragraphs = text.split("\n")
    risk_paragraphs = [p.strip() for p in paragraphs if "risk" in p.lower() and len(p.strip()) > 50]
    collected = "\n\n".join(risk_paragraphs[:20])
    print("🪵 Fallback preview:", repr(collected[:500]))
    return collected if collected else None

def get_risk_section(text):
    section = extract_between_items(text)
    return section if section else fallback_collect_risk_paragraphs(text)
