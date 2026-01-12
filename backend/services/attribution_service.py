"""
Token Attribution Service for Risk Scoring.

Implements perturbation-based attribution to explain which tokens
drive the risk similarity score for any given paragraph.

Methodology: Leave-one-out perturbation
- For each token, measure score change when token is removed
- Positive contribution = removing token decreases score (token increases risk)
- Negative contribution = removing token increases score (token decreases risk)

Deliverable: {"top_tokens": ["regulatory", "material", "adverse"], "contributions": [0.23, 0.18, 0.15]}
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_logger

logger = get_logger(__name__)

# Risk prompt for similarity scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Common stopwords to skip (low attribution expected)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its',
    'they', 'their', 'them', 'we', 'our', 'us', 'you', 'your', 'he', 'she', 'his',
    'her', 'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'also', 'now', 'here', 'there', 'then', 'once', 'if', 'any', 'into', 'over',
    'after', 'before', 'between', 'under', 'above', 'below', 'up', 'down', 'out',
    'off', 'about', 'against', 'during', 'through', 'being', 'having', 'including',
}

# Lazy-loaded model
_model = None
_risk_embedding = None


@dataclass
class TokenAttribution:
    """Attribution result for a single token."""
    token: str
    contribution: float
    position: int
    
    def to_dict(self) -> Dict:
        return {
            "token": self.token,
            "contribution": round(self.contribution, 4),
            "position": self.position,
        }


@dataclass
class AttributionResult:
    """Complete attribution result for a paragraph."""
    text: str
    base_score: float
    top_tokens: List[TokenAttribution]
    all_attributions: List[TokenAttribution]
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "base_score": round(self.base_score, 4),
            "top_tokens": [t.token for t in self.top_tokens],
            "contributions": [round(t.contribution, 4) for t in self.top_tokens],
            "token_details": [t.to_dict() for t in self.top_tokens],
        }


def get_model(model_name: str = "ProsusAI/finbert") -> SentenceTransformer:
    """Get or load the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def get_risk_embedding(model_name: str = "ProsusAI/finbert") -> np.ndarray:
    """Get the cached risk prompt embedding."""
    global _risk_embedding
    if _risk_embedding is None:
        model = get_model(model_name)
        _risk_embedding = model.encode(RISK_PROMPT, convert_to_numpy=True).reshape(1, -1)
    return _risk_embedding


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple word tokenization that preserves positions.
    
    Returns: List of (token, start_pos, end_pos)
    """
    tokens = []
    for match in re.finditer(r'\b\w+\b', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def compute_score(text: str, model: SentenceTransformer, 
                  risk_embedding: np.ndarray) -> float:
    """Compute risk similarity score for a text."""
    if not text.strip():
        return 0.0
    embedding = model.encode(text, convert_to_numpy=True).reshape(1, -1)
    score = cosine_similarity(embedding, risk_embedding)[0, 0]
    return float(np.clip(score, 0, 1))


def compute_token_attributions(
    text: str,
    model_name: str = "ProsusAI/finbert",
    top_n: int = 10,
    include_stopwords: bool = False,
    batch_size: int = 32,
) -> AttributionResult:
    """
    Compute token attributions using leave-one-out perturbation.
    
    Args:
        text: Input paragraph text
        model_name: Sentence transformer model to use
        top_n: Number of top tokens to return
        include_stopwords: Whether to include stopwords in analysis
        batch_size: Batch size for embedding computation
        
    Returns:
        AttributionResult with top contributing tokens
    """
    model = get_model(model_name)
    risk_embedding = get_risk_embedding(model_name)
    
    # Compute base score
    base_score = compute_score(text, model, risk_embedding)
    logger.info(f"Base score: {base_score:.4f}")
    
    # Tokenize
    tokens = tokenize(text)
    logger.info(f"Found {len(tokens)} tokens")
    
    if not tokens:
        return AttributionResult(
            text=text,
            base_score=base_score,
            top_tokens=[],
            all_attributions=[],
        )
    
    # Filter tokens to analyze
    tokens_to_analyze = []
    for token, start, end in tokens:
        if not include_stopwords and token.lower() in STOPWORDS:
            continue
        if len(token) < 2:  # Skip single characters
            continue
        tokens_to_analyze.append((token, start, end))
    
    logger.info(f"Analyzing {len(tokens_to_analyze)} non-stopword tokens")
    
    # Create perturbed texts (remove each token)
    perturbed_texts = []
    for token, start, end in tokens_to_analyze:
        # Remove token from text
        perturbed = text[:start] + text[end:]
        perturbed = re.sub(r'\s+', ' ', perturbed).strip()  # Clean up whitespace
        perturbed_texts.append(perturbed)
    
    # Batch compute scores for perturbed texts
    attributions = []
    
    for i in range(0, len(perturbed_texts), batch_size):
        batch_texts = perturbed_texts[i:i + batch_size]
        batch_tokens = tokens_to_analyze[i:i + batch_size]
        
        # Encode batch
        if batch_texts:
            embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            
            for j, (token, start, end) in enumerate(batch_tokens):
                perturbed_score = cosine_similarity(
                    embeddings[j:j+1], risk_embedding
                )[0, 0]
                perturbed_score = float(np.clip(perturbed_score, 0, 1))
                
                # Contribution = how much score drops when token is removed
                # Positive = token increases risk score
                contribution = base_score - perturbed_score
                
                attributions.append(TokenAttribution(
                    token=token,
                    contribution=contribution,
                    position=start,
                ))
    
    # Sort by absolute contribution (most influential first)
    attributions.sort(key=lambda x: abs(x.contribution), reverse=True)
    
    # Get top positive contributors (tokens that increase risk)
    top_positive = [a for a in attributions if a.contribution > 0][:top_n]
    
    logger.info(f"Top {len(top_positive)} positive contributors identified")
    
    return AttributionResult(
        text=text,
        base_score=base_score,
        top_tokens=top_positive,
        all_attributions=attributions,
    )


def explain_paragraph(text: str, top_n: int = 10) -> Dict:
    """
    High-level API: Explain which tokens drive the risk score.
    
    Args:
        text: Paragraph to explain
        top_n: Number of top tokens to return
        
    Returns:
        Dict with top_tokens and contributions (the deliverable format)
    """
    result = compute_token_attributions(text, top_n=top_n)
    return result.to_dict()


def highlight_text(text: str, attributions: List[TokenAttribution], 
                   top_n: int = 5) -> str:
    """
    Generate HTML-highlighted version of text with attributed tokens.
    
    High-contribution tokens are highlighted in red intensity based on contribution.
    """
    # Get top tokens
    top_tokens = {a.token.lower(): a.contribution for a in attributions[:top_n]}
    
    if not top_tokens:
        return text
    
    max_contrib = max(top_tokens.values()) if top_tokens else 1
    
    def highlight_token(match):
        token = match.group()
        token_lower = token.lower()
        
        if token_lower in top_tokens:
            contrib = top_tokens[token_lower]
            # Normalize intensity (0.3 to 1.0 for visibility)
            intensity = 0.3 + 0.7 * (contrib / max_contrib) if max_contrib > 0 else 0.5
            # Red highlight for risk-increasing tokens
            return f'<span style="background-color: rgba(255, 0, 0, {intensity:.2f}); padding: 2px 4px; border-radius: 3px;">{token}</span>'
        return token
    
    highlighted = re.sub(r'\b\w+\b', highlight_token, text)
    return highlighted


def get_attribution_summary(text: str, top_n: int = 5) -> str:
    """
    Generate a human-readable attribution summary.
    """
    result = compute_token_attributions(text, top_n=top_n)
    
    lines = [
        f"Risk Score: {result.base_score:.2%}",
        "",
        "Top Risk-Driving Tokens:",
    ]
    
    for i, attr in enumerate(result.top_tokens, 1):
        pct = attr.contribution / result.base_score * 100 if result.base_score > 0 else 0
        lines.append(f"  {i}. \"{attr.token}\" (+{attr.contribution:.4f}, ~{pct:.1f}% of score)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Token attribution for risk scoring")
    parser.add_argument("--text", type=str, help="Text to explain")
    parser.add_argument("--file", type=str, help="File containing text to explain")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top tokens")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    # Get text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        # Demo with sample text
        text = """
        The Company is exposed to significant cybersecurity threats and data breaches 
        that could materially adversely affect our business, financial condition, and 
        results of operations. Regulatory compliance failures could result in substantial 
        litigation costs and reputational damage.
        """
    
    print("\n" + "="*70)
    print("TOKEN ATTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nInput text:\n{text.strip()[:200]}...")
    
    result = compute_token_attributions(text, top_n=args.top_n)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    
    print(f"\nBase risk score: {result.base_score:.4f}")
    
    print(f"\nTop {len(result.top_tokens)} risk-driving tokens:")
    print(f"{'Token':<20} {'Contribution':<15} {'Position':<10}")
    print("-" * 45)
    
    for attr in result.top_tokens:
        print(f"{attr.token:<20} {attr.contribution:>+.4f}        {attr.position}")
    
    # Print deliverable format
    print("\n" + "="*70)
    print("DELIVERABLE FORMAT")
    print("="*70)
    
    output = result.to_dict()
    print(json.dumps({
        "top_tokens": output["top_tokens"],
        "contributions": output["contributions"],
    }, indent=2))
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Full results saved to {args.output}")
    
    # Print highlighted text
    print("\n" + "="*70)
    print("HIGHLIGHTED TEXT (HTML)")
    print("="*70)
    highlighted = highlight_text(text, result.top_tokens, top_n=5)
    print(highlighted[:500] + "..." if len(highlighted) > 500 else highlighted)

