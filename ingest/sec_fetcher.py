from infera_config import SEC_API_KEY, SEC_USER_AGENT
import requests
import os
import sys
import time
import re

# Bootstrap project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from infera_config import SEC_API_KEY



# Optional environment variable fallback
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "infera-bot/1.0 (contact@infera.app)")

BASE_API_URL = "https://api.sec-api.io"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

def fetch_latest_10k(ticker):
    print(f"🔍 Searching for latest 10-K for {ticker}...")

    query_url = f"{BASE_API_URL}?token={SEC_API_KEY}"
    payload = {
        "query": f'formType:"10-K" AND ticker:{ticker}',
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    response = requests.post(query_url, json=payload)
    response.raise_for_status()
    filings = response.json().get("filings", [])

    if not filings:
        print(f"❌ No 10-K filings found for {ticker}")
        return

    filing = filings[0]
    cik = filing.get("cik")
    accession = filing.get("accessionNo")

    acc_no_nodashes = re.sub(r"[-]", "", accession)
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_nodashes}/index.json"

    print(f"📄 Fetching index.json from SEC for CIK {cik}...")
    time.sleep(1)
    idx_response = requests.get(index_url, headers=HEADERS)
    idx_response.raise_for_status()

    doc_list = idx_response.json().get("directory", {}).get("item", [])
    primary_doc = next((d["name"] for d in doc_list if d["name"].endswith(".htm")), None)

    if not primary_doc:
        print("❌ Primary document (10-K HTML) not found.")
        return

    filing_url = f"{SEC_ARCHIVES_BASE}/{cik}/{acc_no_nodashes}/{primary_doc}"
    print(f"📥 Downloading 10-K HTML from: {filing_url}")
    time.sleep(2)

    html_response = requests.get(filing_url, headers=HEADERS)
    html_response.raise_for_status()

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, f"{ticker}_10k.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_response.text)

    print(f"✅ 10-K HTML saved to: {save_path}")

if __name__ == "__main__":
    fetch_latest_10k("AAPL")  # or any ticker you'd like to test
