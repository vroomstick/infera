"""
Evaluation harness for risk scoring models.
Computes metrics comparing model predictions against human labels.
"""

import os
import sys
import json
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import spearmanr

# Import project modules
from services.scoring_service import compute_risk_scores

# Label mapping
LABEL_TO_NUMERIC = {"high": 2, "medium": 1, "low": 0}
NUMERIC_TO_LABEL = {2: "high", 1: "medium", 0: "low"}

# TF-IDF risk prompt (from analyze/scorer.py)
TFIDF_RISK_PROMPT = (
    "lawsuits litigation disruption economic downturn regulation compliance product recalls "
    "cybersecurity natural disaster pandemic war inflation supply chain fraud data breach labor shortage"
)

# Embedding risk prompt (from services/scoring_service.py)
EMBEDDING_RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_with_tfidf(texts: List[str]) -> List[float]:
    """Score texts using TF-IDF + cosine similarity."""
    corpus = [TFIDF_RISK_PROMPT] + texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Cosine similarity between prompt and each text
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities.tolist()


def score_with_embeddings(texts: List[str]) -> List[float]:
    """Score texts using sentence embeddings + cosine similarity."""
    results = compute_risk_scores(texts)
    return [score for score, _ in results]


def score_to_label(score: float, method: str = "embedding") -> str:
    """Convert numeric score to label category."""
    # Thresholds tuned based on typical score distributions
    if method == "embedding":
        if score >= 0.55:
            return "high"
        elif score >= 0.45:
            return "medium"
        else:
            return "low"
    else:  # tfidf
        if score >= 0.15:
            return "high"
        elif score >= 0.08:
            return "medium"
        else:
            return "low"


def compute_metrics(y_true: List[str], y_pred: List[str], scores: List[float]) -> Dict:
    """Compute evaluation metrics."""
    # Convert to numeric for some metrics
    y_true_num = [LABEL_TO_NUMERIC[l] for l in y_true]
    y_pred_num = [LABEL_TO_NUMERIC[l] for l in y_pred]
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["low", "medium", "high"])
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=["low", "medium", "high"], output_dict=True)
    
    # Spearman correlation (do scores preserve ranking?)
    spearman_corr, spearman_p = spearmanr(y_true_num, scores)
    
    # Precision@k for high-risk retrieval
    # Sort by score descending, check how many of top-k are actually high
    sorted_indices = np.argsort(scores)[::-1]
    
    precision_at_5 = sum(1 for i in sorted_indices[:5] if y_true[i] == "high") / 5
    precision_at_10 = sum(1 for i in sorted_indices[:10] if y_true[i] == "high") / 10
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "precision_at_5": precision_at_5,
        "precision_at_10": precision_at_10,
    }


def evaluate_method(samples: List[Dict], method: str = "embedding") -> Dict:
    """Evaluate a scoring method on the labeled dataset."""
    texts = [s["text"] for s in samples]
    y_true = [s["label"] for s in samples]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {method.upper()}")
    print(f"{'='*60}")
    
    # Get scores
    if method == "embedding":
        scores = score_with_embeddings(texts)
    else:
        scores = score_with_tfidf(texts)
    
    # Convert scores to labels
    y_pred = [score_to_label(s, method) for s in scores]
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, scores)
    
    # Print results
    print(f"\nAccuracy: {metrics['accuracy']:.1%}")
    print(f"Spearman ρ: {metrics['spearman_correlation']:.3f} (p={metrics['spearman_p_value']:.4f})")
    print(f"Precision@5: {metrics['precision_at_5']:.1%}")
    print(f"Precision@10: {metrics['precision_at_10']:.1%}")
    
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"           low  med  high")
    labels = ["low", "med", "high"]
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {labels[i]:>4}  {row}")
    
    print(f"\nPer-class metrics:")
    for label in ["low", "medium", "high"]:
        r = metrics["classification_report"][label]
        print(f"  {label}: precision={r['precision']:.2f}, recall={r['recall']:.2f}, f1={r['f1-score']:.2f}")
    
    # Add raw predictions for analysis
    metrics["predictions"] = [
        {
            "id": s["id"],
            "source": s["source"],
            "true_label": y_true[i],
            "pred_label": y_pred[i],
            "score": round(scores[i], 4),
            "correct": y_true[i] == y_pred[i],
            "text_preview": s["text"][:100] + "..."
        }
        for i, s in enumerate(samples)
    ]
    
    return metrics


def run_evaluation(data_path: str = "evaluation/labeled_risks.json") -> Dict:
    """Run full evaluation and return results."""
    print("\n" + "="*60)
    print("INFERA RISK SCORING EVALUATION")
    print("="*60)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    
    print(f"\nLoaded {len(samples)} labeled samples")
    print(f"Distribution: {data['summary']['distribution']}")
    
    # Evaluate both methods
    results = {
        "embedding": evaluate_method(samples, "embedding"),
        "tfidf": evaluate_method(samples, "tfidf"),
    }
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<25} {'TF-IDF':<15} {'Embeddings':<15} {'Winner':<10}")
    print("-" * 65)
    
    metrics_to_compare = [
        ("Accuracy", "accuracy", "{:.1%}"),
        ("Spearman ρ", "spearman_correlation", "{:.3f}"),
        ("Precision@5", "precision_at_5", "{:.1%}"),
        ("Precision@10", "precision_at_10", "{:.1%}"),
    ]
    
    for name, key, fmt in metrics_to_compare:
        tfidf_val = results["tfidf"][key]
        emb_val = results["embedding"][key]
        winner = "Embedding" if emb_val > tfidf_val else "TF-IDF" if tfidf_val > emb_val else "Tie"
        print(f"{name:<25} {fmt.format(tfidf_val):<15} {fmt.format(emb_val):<15} {winner:<10}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate risk scoring models")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/eval_results.json", help="Output path for results")
    args = parser.parse_args()
    
    results = run_evaluation(args.data)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {args.output}")

