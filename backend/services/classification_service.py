"""
Risk Taxonomy Classification Service (Phase 1.6)

Classifies risk paragraphs into predefined categories using embedding similarity.

Categories:
- Cybersecurity: data breaches, hacking, malware, cyber attacks
- Regulatory: government regulations, legal compliance
- Supply Chain: suppliers, manufacturing, logistics
- Financial: credit risk, liquidity, interest rates, currency
- Competitive: competition, market share, pricing pressure
- Operational: operations, processes, execution, workforce
- Macroeconomic: economic conditions, recession, inflation
- Litigation: lawsuits, legal proceedings, claims

Output: Each paragraph tagged with category + confidence score
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_logger

logger = get_logger(__name__)

# Risk category prototypes with descriptive text for embedding
RISK_CATEGORIES = {
    "Cybersecurity": (
        "Cybersecurity risks including data breaches, hacking, malware, ransomware, "
        "cyber attacks, unauthorized access to systems, theft of confidential information, "
        "network security vulnerabilities, phishing attacks, and IT system failures."
    ),
    "Regulatory": (
        "Regulatory and compliance risks including government regulations, legal requirements, "
        "regulatory changes, compliance failures, permits and licenses, antitrust laws, "
        "environmental regulations, tax regulations, and industry-specific rules."
    ),
    "Supply Chain": (
        "Supply chain risks including supplier disruptions, manufacturing delays, "
        "logistics problems, component shortages, raw material availability, "
        "transportation issues, inventory management, and sourcing concentration."
    ),
    "Financial": (
        "Financial risks including credit risk, liquidity problems, interest rate exposure, "
        "foreign currency fluctuations, capital market access, debt covenants, "
        "counterparty risk, and cash flow volatility."
    ),
    "Competitive": (
        "Competitive risks including market competition, pricing pressure, new market entrants, "
        "market share loss, technology disruption, product obsolescence, "
        "customer preference changes, and competitive disadvantage."
    ),
    "Operational": (
        "Operational risks including business operations, process failures, execution problems, "
        "workforce challenges, labor shortages, employee retention, productivity issues, "
        "quality control, and operational inefficiencies."
    ),
    "Macroeconomic": (
        "Macroeconomic risks including economic conditions, recession, inflation, "
        "consumer spending decline, GDP contraction, unemployment, market volatility, "
        "trade policies, tariffs, and global economic uncertainty."
    ),
    "Litigation": (
        "Litigation risks including lawsuits, legal proceedings, claims, disputes, "
        "settlements, class action lawsuits, intellectual property disputes, "
        "product liability, and regulatory enforcement actions."
    ),
}

# Lazy-loaded model and category embeddings
_model = None
_category_embeddings = None
_category_names = None


@dataclass
class ClassificationResult:
    """Classification result for a single paragraph."""
    category: str
    confidence: float
    all_scores: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


@dataclass
class BatchClassificationResult:
    """Classification results for multiple paragraphs."""
    results: List[ClassificationResult]
    category_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "category_distribution": self.category_distribution,
        }


def get_model(model_name: str = "ProsusAI/finbert") -> SentenceTransformer:
    """Get or load the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading classification model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def get_category_embeddings(model_name: str = "ProsusAI/finbert") -> Tuple[np.ndarray, List[str]]:
    """
    Get cached category prototype embeddings.
    
    Returns: (embeddings array, category names list)
    """
    global _category_embeddings, _category_names
    
    if _category_embeddings is None:
        model = get_model(model_name)
        
        _category_names = list(RISK_CATEGORIES.keys())
        category_texts = list(RISK_CATEGORIES.values())
        
        logger.info(f"Computing embeddings for {len(_category_names)} category prototypes")
        _category_embeddings = model.encode(category_texts, convert_to_numpy=True)
    
    return _category_embeddings, _category_names


def classify_paragraph(
    text: str,
    model_name: str = "ProsusAI/finbert",
    return_all_scores: bool = True,
) -> ClassificationResult:
    """
    Classify a single paragraph into a risk category.
    
    Args:
        text: Paragraph text to classify
        model_name: Sentence transformer model to use
        return_all_scores: Whether to include all category scores
        
    Returns:
        ClassificationResult with category, confidence, and optionally all scores
    """
    model = get_model(model_name)
    category_embeddings, category_names = get_category_embeddings(model_name)
    
    # Embed the paragraph
    text_embedding = model.encode(text, convert_to_numpy=True).reshape(1, -1)
    
    # Compute similarity to all categories
    similarities = cosine_similarity(text_embedding, category_embeddings)[0]
    
    # Get top category
    top_idx = np.argmax(similarities)
    top_category = category_names[top_idx]
    top_score = float(similarities[top_idx])
    
    # Compute confidence as the gap between top and second-best
    sorted_scores = np.sort(similarities)[::-1]
    confidence = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else top_score
    
    # Normalize confidence to 0-1 range (max gap is ~0.3 typically)
    confidence = min(confidence * 3, 1.0)
    
    all_scores = {}
    if return_all_scores:
        all_scores = {name: float(sim) for name, sim in zip(category_names, similarities)}
    
    return ClassificationResult(
        category=top_category,
        confidence=confidence,
        all_scores=all_scores,
    )


def classify_paragraphs(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
) -> BatchClassificationResult:
    """
    Classify multiple paragraphs into risk categories.
    
    Args:
        texts: List of paragraph texts
        model_name: Sentence transformer model to use
        batch_size: Batch size for embedding computation
        
    Returns:
        BatchClassificationResult with all results and distribution
    """
    model = get_model(model_name)
    category_embeddings, category_names = get_category_embeddings(model_name)
    
    logger.info(f"Classifying {len(texts)} paragraphs")
    
    # Embed all texts
    text_embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        batch_size=batch_size,
        show_progress_bar=len(texts) > 10,
    )
    
    # Compute similarities to all categories
    all_similarities = cosine_similarity(text_embeddings, category_embeddings)
    
    # Build results
    results = []
    category_counts = {name: 0 for name in category_names}
    
    for i, similarities in enumerate(all_similarities):
        top_idx = np.argmax(similarities)
        top_category = category_names[top_idx]
        top_score = float(similarities[top_idx])
        
        # Confidence based on margin
        sorted_scores = np.sort(similarities)[::-1]
        confidence = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else top_score
        confidence = min(confidence * 3, 1.0)
        
        all_scores = {name: float(sim) for name, sim in zip(category_names, similarities)}
        
        results.append(ClassificationResult(
            category=top_category,
            confidence=confidence,
            all_scores=all_scores,
        ))
        
        category_counts[top_category] += 1
    
    return BatchClassificationResult(
        results=results,
        category_distribution=category_counts,
    )


def get_category_prototypes() -> Dict[str, str]:
    """Return the category prototype descriptions."""
    return RISK_CATEGORIES.copy()


def add_category(name: str, description: str) -> None:
    """
    Add a custom category to the taxonomy.
    Note: This clears the cached embeddings.
    """
    global _category_embeddings, _category_names
    
    RISK_CATEGORIES[name] = description
    _category_embeddings = None
    _category_names = None
    
    logger.info(f"Added category: {name}")


def get_top_paragraphs_by_category(
    texts: List[str],
    category: str,
    top_n: int = 5,
    model_name: str = "ProsusAI/finbert",
) -> List[Tuple[int, float, str]]:
    """
    Get the top N paragraphs for a specific category.
    
    Returns: List of (index, score, text) tuples
    """
    model = get_model(model_name)
    category_embeddings, category_names = get_category_embeddings(model_name)
    
    if category not in category_names:
        raise ValueError(f"Unknown category: {category}. Valid: {category_names}")
    
    cat_idx = category_names.index(category)
    cat_embedding = category_embeddings[cat_idx:cat_idx+1]
    
    # Embed texts
    text_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Compute similarities
    similarities = cosine_similarity(text_embeddings, cat_embedding).flatten()
    
    # Get top N
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    return [(int(i), float(similarities[i]), texts[i]) for i in top_indices]


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Risk taxonomy classification")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="JSON file with texts to classify")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--eval", action="store_true", help="Evaluate on labeled dataset")
    args = parser.parse_args()
    
    if args.text:
        # Single text classification
        result = classify_paragraph(args.text)
        print("\n" + "="*60)
        print("CLASSIFICATION RESULT")
        print("="*60)
        print(f"\nCategory: {result.category}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"\nAll scores:")
        for cat, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"  {cat:<15}: {score:.3f} {bar}")
    
    elif args.eval:
        # Evaluate on labeled dataset
        print("\n" + "="*70)
        print("RISK TAXONOMY CLASSIFICATION EVALUATION")
        print("="*70)
        
        # Load labeled data
        with open("evaluation/labeled_risks.json", "r") as f:
            data = json.load(f)
        
        samples = data["samples"]
        texts = [s["text"] for s in samples]
        true_labels = [s["label"] for s in samples]
        sources = [s["source"] for s in samples]
        
        # Classify all
        result = classify_paragraphs(texts)
        
        print(f"\nProcessed {len(texts)} paragraphs")
        print(f"\nCategory distribution:")
        for cat, count in sorted(result.category_distribution.items(), key=lambda x: -x[1]):
            pct = count / len(texts) * 100
            bar = "█" * int(pct / 2)
            print(f"  {cat:<15}: {count:>3} ({pct:>5.1f}%) {bar}")
        
        # Cross-tabulate with severity labels
        print(f"\nCategory vs Severity:")
        print(f"{'Category':<15} {'High':>6} {'Med':>6} {'Low':>6}")
        print("-" * 40)
        
        from collections import defaultdict
        cat_label_counts = defaultdict(lambda: {"high": 0, "medium": 0, "low": 0})
        
        for r, label in zip(result.results, true_labels):
            cat_label_counts[r.category][label] += 1
        
        for cat in sorted(cat_label_counts.keys()):
            counts = cat_label_counts[cat]
            print(f"{cat:<15} {counts['high']:>6} {counts['medium']:>6} {counts['low']:>6}")
        
        # High-confidence examples per category
        print(f"\n\nTop examples per category (highest confidence):")
        
        cat_examples = defaultdict(list)
        for i, r in enumerate(result.results):
            cat_examples[r.category].append((r.confidence, texts[i][:100], sources[i]))
        
        for cat in sorted(cat_examples.keys()):
            examples = sorted(cat_examples[cat], key=lambda x: -x[0])[:2]
            print(f"\n{cat}:")
            for conf, text, source in examples:
                print(f"  [{source}] (conf: {conf:.2f}) {text}...")
        
        # Save results
        if args.output:
            output = {
                "category_distribution": result.category_distribution,
                "detailed_results": [
                    {
                        "id": samples[i]["id"],
                        "source": sources[i],
                        "severity_label": true_labels[i],
                        "category": r.category,
                        "category_confidence": r.confidence,
                        "all_category_scores": r.all_scores,
                    }
                    for i, r in enumerate(result.results)
                ]
            }
            
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\n✅ Results saved to {args.output}")
    
    else:
        # Demo with sample texts
        demo_texts = [
            "The Company is subject to significant cybersecurity threats that could result in data breaches and unauthorized access to confidential information.",
            "Changes in government regulations could require us to modify our business practices and could increase compliance costs.",
            "We rely on a limited number of suppliers for key components, and any disruption could materially impact production.",
            "The Company is exposed to foreign currency exchange rate fluctuations that could adversely affect financial results.",
            "Intense competition from existing players and new market entrants could erode our market share and pricing power.",
        ]
        
        print("\n" + "="*70)
        print("RISK TAXONOMY CLASSIFICATION DEMO")
        print("="*70)
        
        result = classify_paragraphs(demo_texts)
        
        for i, (text, r) in enumerate(zip(demo_texts, result.results), 1):
            print(f"\n{i}. {text[:80]}...")
            print(f"   → Category: {r.category} (confidence: {r.confidence:.1%})")

