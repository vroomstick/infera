# analyze/cleaner.py

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import re
import os

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def clean_html(filepath):
    """
    Clean HTML content from a file and extract readable text.
    
    Args:
        filepath (str): Path to the HTML file to clean
        
    Returns:
        str: Cleaned text content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
        Exception: For other parsing errors
    """
    # Validate file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Validate file is not a directory
    if os.path.isdir(filepath):
        raise ValueError(f"Path is a directory, not a file: {filepath}")
    
    try:
        # Read file with error handling
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {filepath}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Unable to decode file as UTF-8: {filepath}. Error: {e}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {e}")
    
    # Validate file is not empty
    if not html or len(html.strip()) == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    try:
        # Use BeautifulSoup with lxml parser
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        raise Exception(f"Error parsing HTML with BeautifulSoup: {e}")
    
    # Validate soup was created successfully
    if soup is None:
        raise ValueError("BeautifulSoup failed to parse HTML")
    
    try:
        # Remove script, style, nav, and footer tags
        for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
            tag.decompose()

        # Get visible text with newlines
        text = soup.get_text(separator="\n")

        # Remove extra whitespace and non-textual noise
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # remove empty lines
        text_block = "\n".join(lines)

        # Triage: Remove leftover HTML entities (e.g., &nbsp;) and very long garbage lines
        text_block = re.sub(r"&[a-z]+;", " ", text_block)  # replace HTML entities
        text_block = "\n".join([line for line in text_block.splitlines() if len(line) < 500])

        # Triage: sanity check for accidental leftover HTML
        suspicious_lines = [line for line in text_block.splitlines() if "<" in line or ">" in line]
        if suspicious_lines:
            print("⚠️ Triage Warning: Detected potential leftover HTML in cleaned output.")
            print("🔍 Example:", suspicious_lines[:3])

        # Validate we have some content
        if not text_block or len(text_block.strip()) == 0:
            raise ValueError("Cleaned text is empty - file may not contain readable content")

        return text_block

    except Exception as e:
        raise Exception(f"Error processing HTML content: {e}")
