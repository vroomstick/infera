"""
Similarity Failure Mode Analysis (Phase 1.5)

Identifies cases where cosine similarity fails:
1. High score but low actual risk (false positives)
2. Low score but high actual risk (false negatives)

Categorizes failure types:
- Boilerplate: Generic risk language without substance
- Negation: Text describes risk mitigation, not actual risk
- Hedging: Overly cautious language without material concern
- Length bias: Very long or short texts scoring unexpectedly
- Domain mismatch: Financial jargon that confuses embeddings

Output: docs/similarity_failure_modes.md
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Risk prompt for similarity scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Failure type definitions with keywords/patterns for auto-detection
FAILURE_TYPES = {
    "boilerplate": {
        "description": "Generic risk language that appears in every filing without substance",
        "indicators": [
            "factors affecting",
            "from time to time",
            "forward-looking statements",
            "past performance",
            "historical trends",
            "may affect",
            "could be affected",
            "subject to risks",
        ],
    },
    "negation": {
        "description": "Text describes risk mitigation or controls rather than actual risk",
        "indicators": [
            "maintain insurance",
            "insurance coverage",
            "hedging strategies",
            "we hedge",
            "risk management",
            "controls in place",
            "mitigating measures",
            "reduce risk",
            "offset risk",
            "risk mitigation",
        ],
    },
    "hedging_language": {
        "description": "Overly cautious legal language without material concern",
        "indicators": [
            "may or may not",
            "could potentially",
            "there can be no assurance",
            "cannot predict",
            "difficult to predict",
            "subject to change",
            "may vary",
        ],
    },
    "cross_reference": {
        "description": "References other sections without substantive content",
        "indicators": [
            "as discussed in",
            "see part",
            "refer to",
            "incorporated by reference",
            "described elsewhere",
            "as described above",
            "as noted in",
        ],
    },
    "governance": {
        "description": "Corporate governance text mistakenly scored as risk",
        "indicators": [
            "board of directors",
            "audit committee",
            "corporate governance",
            "director independence",
            "committee charter",
            "stockholder rights",
        ],
    },
    "true_positive": {
        "description": "Correctly identified high risk",
        "indicators": [],  # No false positive indicators
    },
    "true_negative": {
        "description": "Correctly identified low risk",
        "indicators": [],
    },
    "missed_severity": {
        "description": "Real risk that embeddings underweighted",
        "indicators": [],  # Requires manual analysis
    },
}


@dataclass
class FailureCase:
    """A single failure case with analysis."""
    id: int
    text: str
    source: str
    true_label: str
    predicted_label: str
    score: float
    failure_type: str
    detected_indicators: List[str] = field(default_factory=list)
    analysis: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "source": self.source,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "score": round(self.score, 4),
            "failure_type": self.failure_type,
            "detected_indicators": self.detected_indicators,
            "analysis": self.analysis,
        }


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_texts(texts: List[str], model_name: str = "ProsusAI/finbert") -> List[float]:
    """Score texts using embedding similarity."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Scoring {len(texts)} texts...")
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    return np.clip(similarities, 0, 1).tolist()


def score_to_label(score: float, thresholds: Dict[str, float] = None) -> str:
    """Convert numeric score to label category."""
    if thresholds is None:
        thresholds = {"high": 0.70, "medium": 0.65}  # FinBERT optimal thresholds
    
    if score >= thresholds["high"]:
        return "high"
    elif score >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def detect_failure_type(text: str, true_label: str, pred_label: str, score: float) -> Tuple[str, List[str]]:
    """
    Automatically detect the failure type based on text patterns.
    
    Returns: (failure_type, detected_indicators)
    """
    text_lower = text.lower()
    
    # First check if it's actually correct
    if true_label == pred_label:
        if true_label == "high":
            return "true_positive", []
        else:
            return "true_negative", []
    
    # False positive: predicted high but actually low/medium
    if pred_label == "high" and true_label in ["low", "medium"]:
        # Check for boilerplate
        for indicator in FAILURE_TYPES["boilerplate"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["boilerplate"]["indicators"] if i in text_lower]
                return "boilerplate", matched
        
        # Check for negation/mitigation
        for indicator in FAILURE_TYPES["negation"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["negation"]["indicators"] if i in text_lower]
                return "negation", matched
        
        # Check for hedging language
        for indicator in FAILURE_TYPES["hedging_language"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["hedging_language"]["indicators"] if i in text_lower]
                return "hedging_language", matched
        
        # Check for cross-references
        for indicator in FAILURE_TYPES["cross_reference"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["cross_reference"]["indicators"] if i in text_lower]
                return "cross_reference", matched
        
        # Check for governance
        for indicator in FAILURE_TYPES["governance"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["governance"]["indicators"] if i in text_lower]
                return "governance", matched
        
        # Check for length bias
        if len(text) < 100:
            return "short_text_bias", [f"text length: {len(text)} chars"]
        if len(text) > 800:
            return "long_text_bias", [f"text length: {len(text)} chars"]
        
        return "unknown_false_positive", []
    
    # False negative: predicted low but actually high
    if pred_label == "low" and true_label == "high":
        # Check for negation that obscures risk
        for indicator in FAILURE_TYPES["negation"]["indicators"]:
            if indicator in text_lower:
                matched = [i for i in FAILURE_TYPES["negation"]["indicators"] if i in text_lower]
                return "negation_obscures_risk", matched
        
        return "missed_severity", []
    
    # Medium misclassification (softer error)
    if abs(["low", "medium", "high"].index(pred_label) - ["low", "medium", "high"].index(true_label)) == 1:
        return "boundary_error", [f"predicted: {pred_label}, true: {true_label}"]
    
    return "other", []


def analyze_failures(
    data_path: str = "evaluation/labeled_risks.json",
    thresholds: Dict[str, float] = None,
) -> Dict:
    """
    Analyze all failure cases and categorize them.
    """
    if thresholds is None:
        thresholds = {"high": 0.70, "medium": 0.65}
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    
    texts = [s["text"] for s in samples]
    true_labels = [s["label"] for s in samples]
    
    # Score all texts
    scores = score_texts(texts)
    pred_labels = [score_to_label(s, thresholds) for s in scores]
    
    # Analyze each sample
    failure_cases = []
    correct_cases = []
    
    for i, (sample, score, pred_label) in enumerate(zip(samples, scores, pred_labels)):
        true_label = sample["label"]
        failure_type, indicators = detect_failure_type(
            sample["text"], true_label, pred_label, score
        )
        
        case = FailureCase(
            id=sample["id"],
            text=sample["text"],
            source=sample["source"],
            true_label=true_label,
            predicted_label=pred_label,
            score=score,
            failure_type=failure_type,
            detected_indicators=indicators,
            analysis=sample.get("notes", ""),
        )
        
        if true_label == pred_label:
            correct_cases.append(case)
        else:
            failure_cases.append(case)
    
    # Group failures by type
    failures_by_type = defaultdict(list)
    for case in failure_cases:
        failures_by_type[case.failure_type].append(case)
    
    # Sort failures by score (most egregious first)
    for ftype in failures_by_type:
        if "false_positive" in ftype or ftype in ["boilerplate", "negation", "hedging_language"]:
            # High score but wrong - sort by highest score first
            failures_by_type[ftype].sort(key=lambda x: -x.score)
        else:
            # Low score but should be high - sort by lowest score first
            failures_by_type[ftype].sort(key=lambda x: x.score)
    
    # Compute statistics
    total = len(samples)
    correct = len(correct_cases)
    accuracy = correct / total
    
    failure_type_counts = {ftype: len(cases) for ftype, cases in failures_by_type.items()}
    
    print(f"\n{'='*70}")
    print("FAILURE MODE ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal samples: {total}")
    print(f"Correct predictions: {correct} ({accuracy:.1%})")
    print(f"Failures: {len(failure_cases)} ({1-accuracy:.1%})")
    
    print(f"\nFailure breakdown by type:")
    for ftype, count in sorted(failure_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(failure_cases) * 100 if failure_cases else 0
        desc = FAILURE_TYPES.get(ftype, {}).get("description", "")
        print(f"  {ftype:<25}: {count:>3} ({pct:>5.1f}%) - {desc[:50]}")
    
    return {
        "summary": {
            "total_samples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "failure_count": len(failure_cases),
            "thresholds": thresholds,
        },
        "failure_type_counts": failure_type_counts,
        "failure_type_definitions": {
            ftype: info["description"] 
            for ftype, info in FAILURE_TYPES.items()
        },
        "failures_by_type": {
            ftype: [c.to_dict() for c in cases[:5]]  # Top 5 per type
            for ftype, cases in failures_by_type.items()
        },
        "all_failures": [c.to_dict() for c in failure_cases],
    }


def generate_markdown_report(analysis: Dict, output_path: str = "docs/similarity_failure_modes.md"):
    """Generate a markdown report of failure modes."""
    
    lines = [
        "# Similarity Failure Mode Analysis",
        "",
        "This document identifies and categorizes cases where cosine similarity fails for risk scoring.",
        "",
        "## Summary",
        "",
        f"- **Total samples evaluated:** {analysis['summary']['total_samples']}",
        f"- **Accuracy:** {analysis['summary']['accuracy']:.1%}",
        f"- **Total failures:** {analysis['summary']['failure_count']}",
        f"- **Thresholds:** high ≥ {analysis['summary']['thresholds']['high']}, medium ≥ {analysis['summary']['thresholds']['medium']}",
        "",
        "---",
        "",
        "## Failure Type Breakdown",
        "",
        "| Failure Type | Count | % of Failures | Description |",
        "|--------------|-------|---------------|-------------|",
    ]
    
    # Sort by count
    sorted_types = sorted(
        analysis["failure_type_counts"].items(), 
        key=lambda x: -x[1]
    )
    
    total_failures = analysis["summary"]["failure_count"]
    for ftype, count in sorted_types:
        pct = count / total_failures * 100 if total_failures > 0 else 0
        desc = analysis["failure_type_definitions"].get(ftype, "")
        lines.append(f"| {ftype} | {count} | {pct:.1f}% | {desc} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Failure Type Details",
        "",
    ])
    
    # Details for each failure type
    for ftype, cases in analysis["failures_by_type"].items():
        if not cases:
            continue
            
        desc = analysis["failure_type_definitions"].get(ftype, "Unknown type")
        lines.extend([
            f"### {ftype.replace('_', ' ').title()}",
            "",
            f"*{desc}*",
            "",
            f"**{len(cases)} examples (showing top 5):**",
            "",
        ])
        
        for i, case in enumerate(cases[:5], 1):
            lines.extend([
                f"#### Example {i}: {case['source']} (ID: {case['id']})",
                "",
                f"- **True label:** {case['true_label']}",
                f"- **Predicted:** {case['predicted_label']}",
                f"- **Score:** {case['score']:.4f}",
            ])
            
            if case["detected_indicators"]:
                lines.append(f"- **Indicators found:** {', '.join(case['detected_indicators'][:3])}")
            
            lines.extend([
                "",
                "> " + case["text"].replace("\n", " ")[:300] + "...",
                "",
            ])
    
    lines.extend([
        "---",
        "",
        "## Key Insights",
        "",
        "### When Cosine Similarity Lies",
        "",
        "1. **Boilerplate text**: Standard legal language that appears in every 10-K scores high because it contains risk-related words, but conveys no specific threat.",
        "",
        "2. **Risk mitigation descriptions**: Text about insurance, hedging, or controls gets scored as high-risk because it mentions risks (even though it's describing protection).",
        "",
        "3. **Hedging language**: Legal disclaimers using words like 'may', 'could', 'potential' trigger high similarity to the risk prompt.",
        "",
        "4. **Boundary cases**: Medium vs Low and Medium vs High distinctions are inherently fuzzy—cosine similarity provides a continuous score that must be discretized.",
        "",
        "### Recommendations",
        "",
        "1. **Add boilerplate filter**: Pre-filter common boilerplate phrases before scoring.",
        "",
        "2. **Negation detection**: Implement simple negation/mitigation detection to downweight risk mitigation text.",
        "",
        "3. **Calibrate thresholds**: Use probability calibration to convert raw scores to risk probabilities.",
        "",
        "4. **Ensemble approach**: Combine embedding similarity with keyword rules for edge cases.",
        "",
        "---",
        "",
        "*Generated by `evaluation/failure_mode_analysis.py`*",
    ])
    
    # Write markdown
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Markdown report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze similarity failure modes")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/failure_mode_analysis.json", help="Output JSON path")
    parser.add_argument("--markdown", default="docs/similarity_failure_modes.md", help="Output markdown path")
    parser.add_argument("--high-thresh", type=float, default=0.70, help="High risk threshold")
    parser.add_argument("--med-thresh", type=float, default=0.65, help="Medium risk threshold")
    args = parser.parse_args()
    
    thresholds = {"high": args.high_thresh, "medium": args.med_thresh}
    
    # Run analysis
    analysis = analyze_failures(args.data, thresholds)
    
    # Save JSON results
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ JSON results saved to: {args.output}")
    
    # Generate markdown report
    generate_markdown_report(analysis, args.markdown)

