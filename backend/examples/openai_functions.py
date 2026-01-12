"""
OpenAI Function Calling Schema for Infera

Defines function schemas for use with OpenAI's function calling / Assistants API.

Usage with OpenAI API:
    from openai import OpenAI
    from examples.openai_functions import INFERA_FUNCTIONS, execute_function
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What are Apple's main risks?"}],
        tools=INFERA_FUNCTIONS,
        tool_choice="auto"
    )
    
    # Execute the function call
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = execute_function(tool_call.function.name, tool_call.function.arguments)

Usage with Assistants API:
    assistant = client.beta.assistants.create(
        name="Risk Analyst",
        instructions="You analyze SEC filings using Infera tools.",
        model="gpt-4o",
        tools=INFERA_FUNCTIONS
    )
"""

import os
import sys
import json
from typing import Dict, Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# === Function Schemas ===

INFERA_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_risks",
            "description": "Analyze 10-K risk factors for a company. Fetches the filing from SEC EDGAR, extracts risk factors, and scores each paragraph using FinBERT embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT, NVDA)"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Fiscal year (e.g., 2023). If omitted, uses most recent filing."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_risk_score",
            "description": "Explain why a specific paragraph received its risk score. Returns token-level attributions showing which words drive the score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paragraph_id": {
                        "type": "integer",
                        "description": "The database ID of the paragraph to explain"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top contributing tokens to show (default 5)",
                        "default": 5
                    }
                },
                "required": ["paragraph_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_risks",
            "description": "Search for risk paragraphs matching a natural language query using semantic search with FinBERT embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'cybersecurity data breach', 'supply chain disruption')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5)",
                        "default": 5
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional filter by company ticker symbol"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "Compare risk profiles across multiple companies. Provides comparative analysis including risk distributions and rankings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "string",
                        "description": "Comma-separated list of ticker symbols (e.g., 'AAPL,TSLA,MSFT')"
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing_summary",
            "description": "Get a summary of an analyzed company's risk filing including paragraph count, top risks, and availability of GPT summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paragraph_details",
            "description": "Get detailed information about a specific paragraph including text, score, confidence, and optional embedding vector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paragraph_id": {
                        "type": "integer",
                        "description": "The database ID of the paragraph"
                    },
                    "include_embedding": {
                        "type": "boolean",
                        "description": "Whether to include the raw embedding vector (default false)",
                        "default": False
                    }
                },
                "required": ["paragraph_id"]
            }
        }
    }
]


# === Function Implementations ===

def _analyze_risks(ticker: str, year: int = None) -> Dict[str, Any]:
    """Analyze risks for a company."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        result = client.fetch(ticker=ticker, year=year, analyze=True, skip_summary=True)
        return {
            "status": "success",
            "ticker": ticker,
            "filing_id": result.get("filing_id"),
            "paragraph_count": result.get("paragraph_count"),
            "message": result.get("message"),
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


def _explain_risk_score(paragraph_id: int, top_n: int = 5) -> Dict[str, Any]:
    """Explain a risk score."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        explanation = client.explain(paragraph_id=paragraph_id, top_n=top_n)
        return {
            "status": "success",
            "paragraph_id": explanation.paragraph_id,
            "score": explanation.score,
            "confidence": explanation.confidence,
            "risk_category": explanation.risk_category,
            "top_tokens": [
                {"token": t.token, "contribution": t.contribution}
                for t in explanation.top_tokens
            ],
            "text_preview": explanation.text[:300],
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


def _search_risks(query: str, limit: int = 5, ticker: str = None) -> Dict[str, Any]:
    """Search for risks."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        results = client.search(query=query, limit=limit, ticker=ticker)
        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": [
                {
                    "paragraph_id": r.paragraph_id,
                    "ticker": r.ticker,
                    "similarity": r.score,
                    "text_preview": r.text[:200],
                }
                for r in results
            ],
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


def _compare_companies(tickers: str) -> Dict[str, Any]:
    """Compare companies."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        result = client.benchmark(ticker_list)
        return {
            "status": "success",
            "companies": ticker_list,
            "profiles": result.get("profiles", {}),
            "comparison": result.get("comparison", {}),
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


def _get_filing_summary(ticker: str) -> Dict[str, Any]:
    """Get filing summary."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        filings = client.list_filings()
        matching = [f for f in filings if f.ticker.upper() == ticker.upper()]
        
        if not matching:
            return {"status": "not_found", "message": f"No filings for {ticker}"}
        
        latest = matching[0]
        details = client.get_filing(latest.id)
        
        return {
            "status": "success",
            "ticker": ticker,
            "filing_id": latest.id,
            "paragraph_count": details.get("paragraph_count", 0),
            "top_risks": details.get("top_risks", [])[:3],
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


def _get_paragraph_details(paragraph_id: int, include_embedding: bool = False) -> Dict[str, Any]:
    """Get paragraph details."""
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        para = client.get_paragraph(paragraph_id, include_embedding=include_embedding)
        result = {
            "status": "success",
            "paragraph_id": para.paragraph_id,
            "score": para.score,
            "confidence": para.confidence,
            "text": para.text,
        }
        if include_embedding and para.embedding:
            result["embedding_length"] = len(para.embedding)
            result["embedding_preview"] = para.embedding[:5]
        return result
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


# Function registry
FUNCTION_REGISTRY: Dict[str, Callable] = {
    "analyze_risks": _analyze_risks,
    "explain_risk_score": _explain_risk_score,
    "search_risks": _search_risks,
    "compare_companies": _compare_companies,
    "get_filing_summary": _get_filing_summary,
    "get_paragraph_details": _get_paragraph_details,
}


def execute_function(name: str, arguments: str) -> Dict[str, Any]:
    """
    Execute a function by name with JSON arguments.
    
    Args:
        name: Function name from the tool call
        arguments: JSON string of arguments
    
    Returns:
        Function result as dictionary
    """
    if name not in FUNCTION_REGISTRY:
        return {"status": "error", "error": f"Unknown function: {name}"}
    
    try:
        args = json.loads(arguments)
        return FUNCTION_REGISTRY[name](**args)
    except json.JSONDecodeError as e:
        return {"status": "error", "error": f"Invalid arguments JSON: {e}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# === Demo ===

if __name__ == "__main__":
    print("=" * 60)
    print("INFERA OPENAI FUNCTION SCHEMAS")
    print("=" * 60)
    
    print("\n📦 Available functions:")
    for func in INFERA_FUNCTIONS:
        name = func["function"]["name"]
        desc = func["function"]["description"][:50]
        print(f"  • {name}: {desc}...")
    
    print("\n📋 Schema (for OpenAI API):")
    print(json.dumps(INFERA_FUNCTIONS[0], indent=2))
    
    print("\n🔧 Usage example:")
    print("""
    from openai import OpenAI
    from examples.openai_functions import INFERA_FUNCTIONS, execute_function
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What are Tesla's main risks?"}],
        tools=INFERA_FUNCTIONS,
        tool_choice="auto"
    )
    
    if response.choices[0].message.tool_calls:
        for call in response.choices[0].message.tool_calls:
            result = execute_function(call.function.name, call.function.arguments)
            print(result)
    """)

