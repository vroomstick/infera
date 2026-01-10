# analyze/segmenter.py
"""
SEC 10-K Risk Factors section extraction.

Handles multiple 10-K format variations:
- Standard SEC format with Item 1A/1B headers
- Various whitespace and punctuation patterns
- Unicode non-breaking spaces (\xa0)
- Table of contents vs actual section headers
"""

import re
from typing import Optional, List, Tuple


# Patterns for finding Item 1A Risk Factors section start
# Order matters: more specific patterns first
ITEM_1A_PATTERNS = [
    # Standard format with various separators (including \xa0 non-breaking space)
    r"item[\s\xa0]*1a[\s\xa0.:–\-]*[\s\xa0]*risk[\s\xa0]*factors",
    # All caps version
    r"ITEM[\s\xa0]*1A[\s\xa0.:–\-]*[\s\xa0]*RISK[\s\xa0]*FACTORS",
    # Item 1A on its own line, Risk Factors on next
    r"item[\s\xa0]*1a\.?[\s\xa0]*\n+[\s\xa0]*risk[\s\xa0]*factors",
    # Just "Item 1A." followed by any text (for section headers)
    r"item[\s\xa0]*1a\.[\s\xa0]+",
]

# Patterns for finding section end (Item 1B or Item 2)
END_PATTERNS = [
    r"item[\s\xa0]*1b[\s\xa0.:–\-]*[\s\xa0]*unresolved",
    r"item[\s\xa0]*1b\.?[\s\xa0]*\n",
    r"item[\s\xa0]*1b\.[\s\xa0]+",
    r"ITEM[\s\xa0]*1B",
    r"item[\s\xa0]*2[\s\xa0.:–\-]*[\s\xa0]*properties",
    r"ITEM[\s\xa0]*2",
]


def normalize_text(text: str) -> str:
    """Normalize unicode whitespace for consistent matching."""
    # Replace non-breaking spaces with regular spaces for pattern matching
    return text.replace('\xa0', ' ')


def find_section_start(text: str) -> Optional[Tuple[int, int]]:
    """
    Find the start of the Item 1A Risk Factors section.
    
    Returns:
        Tuple of (start_index, end_of_header_index) or None
    """
    # Look for Item 1A followed by Risk Factors with actual content after
    # The key pattern: "Item 1A" + whitespace/punctuation + "Risk Factors" + substantive paragraph
    
    candidates = []
    
    for pattern in ITEM_1A_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for match in regex.finditer(text):
            # Get context after the match
            remaining = text[match.end():match.end() + 500]
            remaining_lines = remaining.split('\n')
            
            # Skip if this is a cross-reference ("as discussed in Part I, Item 1A")
            context_before = text[max(0, match.start() - 100):match.start()]
            if 'discussed in' in context_before.lower() or 'see item' in context_before.lower():
                continue
            
            # Check what follows the header
            # Table of contents entries: followed by page number then next item
            # Actual headers: followed by substantive text
            
            # Look at first few non-empty lines after
            non_empty = [l.strip() for l in remaining_lines if l.strip()][:5]
            
            # If first line after is just a number (page number), this is TOC
            if non_empty and re.match(r'^\d+$', non_empty[0]):
                continue
            
            # If followed immediately by "Item 1B" or similar, this is TOC
            if non_empty and any(re.match(r'^item\s*1b', l, re.IGNORECASE) for l in non_empty[:3]):
                continue
            
            # Score this candidate based on content quality
            # Good sign: followed by long paragraphs
            content_score = sum(len(l) for l in non_empty[:3])
            
            candidates.append((match.start(), match.end(), content_score))
    
    # Pick the candidate with the best content score
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        return (best[0], best[1])
    
    return None


def find_section_end(text: str, start_idx: int) -> int:
    """
    Find the end of the Risk Factors section.
    
    Returns:
        Index where the section ends
    """
    search_text = text[start_idx:]
    search_text_lower = search_text.lower()
    
    # Find the earliest end marker
    earliest_end = len(search_text)
    
    for pattern in END_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        match = regex.search(search_text_lower)
        if match:
            # Make sure it's not just a cross-reference
            context_before = search_text[max(0, match.start() - 50):match.start()]
            if 'see item' in context_before.lower() or 'refer to' in context_before.lower():
                continue
            
            if match.start() < earliest_end and match.start() > 100:  # Must have some content
                earliest_end = match.start()
    
    return start_idx + earliest_end


def extract_between_items(text: str) -> Optional[str]:
    """
    Extract content between Item 1A (Risk Factors) and Item 1B.
    
    Returns:
        Extracted section text or None if not found
    """
    start_result = find_section_start(text)
    
    if not start_result:
        print("❌ Could not locate Item 1A section header.")
        return None
    
    start_idx, header_end = start_result
    end_idx = find_section_end(text, header_end)
    
    # Extract the section content (skip the header itself)
    section = text[header_end:end_idx].strip()
    
    # Validate we have substantial content
    word_count = len(section.split())
    if word_count < 100:
        print(f"⚠️ Primary match too short (length: {word_count} words). Fallback triggered.")
        return None
    
    print(f"✅ Primary extraction (Item 1A → 1B) succeeded. {word_count} words extracted.")
    print("🪵 Preview:", repr(section[:300]))
    return section


def fallback_collect_risk_paragraphs(text: str) -> Optional[str]:
    """
    Fallback: Collect paragraphs containing risk-related keywords.
    
    Used when primary extraction fails.
    """
    print("⚠️ Primary failed. Fallback: collecting risk-related paragraphs.")
    
    # Risk-related keywords
    risk_keywords = [
        'risk', 'adverse', 'material', 'significant', 'litigation',
        'lawsuit', 'regulatory', 'compliance', 'cybersecurity',
        'disruption', 'volatility', 'uncertainty'
    ]
    
    paragraphs = text.split("\n")
    risk_paragraphs = []
    
    for p in paragraphs:
        p_stripped = p.strip()
        p_lower = p_stripped.lower()
        
        # Must be substantial paragraph (>50 chars)
        if len(p_stripped) < 50:
            continue
        
        # Must contain risk-related keywords
        if any(kw in p_lower for kw in risk_keywords):
            risk_paragraphs.append(p_stripped)
    
    # Limit to first 25 paragraphs to avoid too much noise
    collected = "\n\n".join(risk_paragraphs[:25])
    
    if collected:
        print(f"🪵 Fallback collected {len(risk_paragraphs[:25])} paragraphs.")
        print("🪵 Fallback preview:", repr(collected[:300]))
        return collected
    
    return None


def get_risk_section(text: str) -> Optional[str]:
    """
    Main entry point: Extract the Risk Factors section from cleaned 10-K text.
    
    Tries primary extraction first, falls back to keyword-based collection.
    
    Args:
        text: Cleaned text from a 10-K filing
        
    Returns:
        Extracted risk factors section or None
    """
    section = extract_between_items(text)
    return section if section else fallback_collect_risk_paragraphs(text)


# Additional utility functions for multi-format support

def detect_filing_format(text: str) -> str:
    """
    Detect the format/structure of the 10-K filing.
    
    Returns:
        Format identifier string
    """
    text_lower = text.lower()
    
    if 'item 1a' in text_lower and 'item 1b' in text_lower:
        return 'standard_sec'
    elif 'risk factors' in text_lower:
        return 'has_risk_section'
    else:
        return 'unknown'


def extract_all_item_locations(text: str) -> List[Tuple[str, int]]:
    """
    Find all Item markers in the document for debugging.
    
    Returns:
        List of (item_name, position) tuples
    """
    pattern = r"item[\s\xa0]*(\d+[ab]?)[\s\xa0.:–\-]"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    return [(f"Item {m.group(1)}", m.start()) for m in matches]


if __name__ == "__main__":
    # Test on sample filing
    import sys
    sys.path.insert(0, '.')
    from analyze.cleaner import clean_html
    
    test_files = [
        "data/AAPL_10K.html",
        "data/TSLA_10K.html", 
        "data/MSFT_10K.html",
    ]
    
    for filepath in test_files:
        print(f"\n{'='*60}")
        print(f"Testing: {filepath}")
        print('='*60)
        
        try:
            cleaned = clean_html(filepath)
            result = get_risk_section(cleaned)
            
            if result:
                words = len(result.split())
                print(f"✅ Extracted {words} words")
            else:
                print("❌ Failed to extract section")
        except Exception as e:
            print(f"❌ Error: {e}")
