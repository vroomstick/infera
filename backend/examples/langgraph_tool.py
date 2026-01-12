"""
LangGraph Tool Integration for Infera

Makes Infera available as a tool for LangGraph agents.

Usage:
    from examples.langgraph_tool import infera_tools, create_infera_agent
    
    # Use tools directly
    result = analyze_risks.invoke({"ticker": "AAPL"})
    
    # Or create an agent with Infera tools
    agent = create_infera_agent()
    response = agent.invoke({"messages": [HumanMessage("What are AAPL's top risks?")]})

Requirements:
    pip install langgraph langchain-core langchain-openai
"""

import os
import sys
from typing import Optional, List, Dict, Any

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


# === Tool Definitions ===

@tool
def analyze_risks(ticker: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze 10-K risk factors for a company.
    
    Fetches the company's 10-K filing from SEC EDGAR, extracts risk factors,
    scores each paragraph using FinBERT embeddings, and returns top risks.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT, NVDA)
        year: Optional fiscal year (e.g., 2023). If omitted, uses most recent.
    
    Returns:
        Analysis results including top risks, scores, and summary
    """
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


@tool
def explain_risk_score(paragraph_id: int, top_n: int = 5) -> Dict[str, Any]:
    """
    Explain why a specific paragraph received its risk score.
    
    Returns token-level attributions showing which words drive the score.
    Use this to understand what makes a paragraph high or low risk.
    
    Args:
        paragraph_id: The database ID of the paragraph to explain
        top_n: Number of top contributing tokens to show (default 5)
    
    Returns:
        Explanation with score, confidence, top tokens, and risk category
    """
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        explanation = client.explain(paragraph_id=paragraph_id, top_n=top_n)
        return {
            "status": "success",
            "paragraph_id": explanation.paragraph_id,
            "score": explanation.score,
            "confidence": f"{explanation.confidence:.1%}",
            "risk_category": explanation.risk_category,
            "category_confidence": f"{explanation.category_confidence:.1%}" if explanation.category_confidence else None,
            "top_tokens": [
                {"token": t.token, "contribution": f"+{t.contribution:.4f}"}
                for t in explanation.top_tokens
            ],
            "text_preview": explanation.text[:200] + "..." if len(explanation.text) > 200 else explanation.text,
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


@tool
def search_risks(query: str, limit: int = 5, ticker: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for risk paragraphs matching a natural language query.
    
    Uses semantic search with FinBERT embeddings to find similar risks
    across all analyzed filings.
    
    Args:
        query: Natural language search query (e.g., "cybersecurity data breach")
        limit: Maximum number of results to return (default 5)
        ticker: Optional filter by company ticker
    
    Returns:
        List of matching risk paragraphs with scores and sources
    """
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
                    "similarity": f"{r.score:.2f}",
                    "text_preview": r.text[:150] + "..." if len(r.text) > 150 else r.text,
                }
                for r in results
            ],
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


@tool
def compare_companies(tickers: str) -> Dict[str, Any]:
    """
    Compare risk profiles across multiple companies.
    
    Provides comparative analysis including risk distributions, statistics,
    and rankings across companies.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., "AAPL,TSLA,MSFT")
    
    Returns:
        Comparison with profiles, rankings, and insights
    """
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


@tool
def get_filing_summary(ticker: str) -> Dict[str, Any]:
    """
    Get a summary of an analyzed company's risk filing.
    
    Returns the most recent analysis for a company, including
    paragraph count, word count, and availability of summary.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Filing information and availability status
    """
    from sdk.infera_client import InferaClient, InferaError
    
    client = InferaClient()
    try:
        filings = client.list_filings()
        matching = [f for f in filings if f.ticker.upper() == ticker.upper()]
        
        if not matching:
            return {
                "status": "not_found",
                "message": f"No filings found for {ticker}. Use analyze_risks first.",
            }
        
        latest = matching[0]
        details = client.get_filing(latest.id)
        
        return {
            "status": "success",
            "ticker": ticker,
            "filing_id": latest.id,
            "filing_type": latest.filing_type,
            "filing_date": latest.filing_date,
            "has_summary": latest.has_summary,
            "paragraph_count": details.get("paragraph_count", 0),
            "word_count": details.get("word_count", 0),
            "top_risks": details.get("top_risks", [])[:3],
        }
    except InferaError as e:
        return {"status": "error", "error": str(e)}
    finally:
        client.close()


# === Tool Collection ===

# All Infera tools bundled together
INFERA_TOOLS = [
    analyze_risks,
    explain_risk_score,
    search_risks,
    compare_companies,
    get_filing_summary,
]


def get_tools():
    """Get all Infera tools for use with LangGraph."""
    return INFERA_TOOLS


# === LangGraph Agent Creation ===

def create_infera_agent(model_name: str = "gpt-4o"):
    """
    Create a LangGraph agent with Infera tools.
    
    Args:
        model_name: OpenAI model to use (default gpt-4o)
    
    Returns:
        Compiled LangGraph agent ready to use
        
    Example:
        agent = create_infera_agent()
        response = agent.invoke({
            "messages": [HumanMessage("What are Tesla's main risks?")]
        })
    """
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "LangGraph and langchain-openai required. "
            "Install with: pip install langgraph langchain-openai"
        )
    
    # Create the LLM
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Create agent with tools
    agent = create_react_agent(
        llm,
        tools=INFERA_TOOLS,
        state_modifier=(
            "You are a financial risk analyst assistant. "
            "You have access to Infera, a tool for analyzing SEC 10-K filings. "
            "Use the available tools to answer questions about company risks. "
            "Always cite specific paragraph IDs and scores when discussing risks."
        ),
    )
    
    return agent


# === Demo ===

if __name__ == "__main__":
    print("=" * 60)
    print("INFERA LANGGRAPH TOOLS")
    print("=" * 60)
    
    print("\n📦 Available tools:")
    for tool in INFERA_TOOLS:
        print(f"  • {tool.name}: {tool.description[:60]}...")
    
    print("\n🔧 Example usage:")
    print("""
    from examples.langgraph_tool import create_infera_agent
    from langchain_core.messages import HumanMessage
    
    agent = create_infera_agent()
    response = agent.invoke({
        "messages": [HumanMessage("What are Apple's top cybersecurity risks?")]
    })
    print(response["messages"][-1].content)
    """)
    
    # Quick tool test
    print("\n🧪 Testing search_risks tool...")
    try:
        result = search_risks.invoke({"query": "cybersecurity data breach", "limit": 3})
        if result["status"] == "success":
            print(f"  ✅ Found {result['count']} results")
        else:
            print(f"  ⚠️ {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"  ❌ Test failed: {e}")

