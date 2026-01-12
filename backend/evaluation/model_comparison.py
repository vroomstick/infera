"""
Embedding Model Comparison for Risk Scoring.

Tests multiple embedding models to determine optimal choice for risk detection:
1. all-MiniLM-L6-v2 (current baseline - fast, general purpose)
2. all-mpnet-base-v2 (larger, more accurate)
3. ProsusAI/finbert (domain-specific financial model)

Metrics: Accuracy, Spearman ρ, Precision@k
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# Label mappings
LABEL_TO_NUMERIC = {"high": 2, "medium": 1, "low": 0}

# Risk prompt for similarity scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Models to test
MODELS_TO_TEST = [
    {
        "name": "all-MiniLM-L6-v2",
        "model_id": "all-MiniLM-L6-v2",
        "description": "Current baseline - fast, general purpose (22M params)",
        "thresholds": {"high": 0.55, "medium": 0.45}  # Current thresholds
    },
    {
        "name": "all-mpnet-base-v2",
        "model_id": "all-mpnet-base-v2",
        "description": "Larger, more accurate (110M params)",
        "thresholds": {"high": 0.55, "medium": 0.45}
    },
    {
        "name": "FinBERT",
        "model_id": "ProsusAI/finbert",
        "description": "Domain-specific financial model",
        "thresholds": {"high": 0.60, "medium": 0.50}  # May need different thresholds
    },
]


@dataclass
class ModelResult:
    """Results from evaluating a single model."""
    name: str
    description: str
    accuracy: float
    spearman_rho: float
    spearman_p: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    load_time_ms: float
    inference_time_ms: float
    confusion_matrix: List[List[int]]
    classification_report: Dict
    optimal_thresholds: Dict[str, float]


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_to_label(score: float, thresholds: Dict[str, float]) -> str:
    """Convert numeric score to label category using given thresholds."""
    if score >= thresholds["high"]:
        return "high"
    elif score >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def find_optimal_thresholds(scores: List[float], y_true: List[str]) -> Dict[str, float]:
    """
    Find optimal thresholds by grid search.
    Returns thresholds that maximize accuracy.
    """
    best_acc = 0
    best_thresholds = {"high": 0.55, "medium": 0.45}
    
    # Grid search over threshold combinations
    for high_thresh in np.arange(0.40, 0.75, 0.05):
        for med_thresh in np.arange(0.25, high_thresh, 0.05):
            thresholds = {"high": high_thresh, "medium": med_thresh}
            y_pred = [score_to_label(s, thresholds) for s in scores]
            acc = accuracy_score(y_true, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
    
    return best_thresholds


def compute_metrics(y_true: List[str], y_pred: List[str], scores: List[float]) -> Dict:
    """Compute all evaluation metrics."""
    y_true_num = [LABEL_TO_NUMERIC[l] for l in y_true]
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["low", "medium", "high"])
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=["low", "medium", "high"], output_dict=True)
    
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(y_true_num, scores)
    
    # Precision@k for high-risk retrieval
    sorted_indices = np.argsort(scores)[::-1]
    
    precision_at_5 = sum(1 for i in sorted_indices[:5] if y_true[i] == "high") / 5
    precision_at_10 = sum(1 for i in sorted_indices[:10] if y_true[i] == "high") / 10
    precision_at_20 = sum(1 for i in sorted_indices[:20] if y_true[i] == "high") / 20
    
    return {
        "accuracy": accuracy,
        "spearman_rho": spearman_corr,
        "spearman_p": spearman_p,
        "precision_at_5": precision_at_5,
        "precision_at_10": precision_at_10,
        "precision_at_20": precision_at_20,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_model(model_config: Dict, texts: List[str], y_true: List[str]) -> ModelResult:
    """Evaluate a single embedding model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['name']}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model...")
    load_start = time.time()
    try:
        model = SentenceTransformer(model_config["model_id"])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Return dummy result for failed models
        return ModelResult(
            name=model_config["name"],
            description=f"FAILED: {e}",
            accuracy=0, spearman_rho=0, spearman_p=1,
            precision_at_5=0, precision_at_10=0, precision_at_20=0,
            load_time_ms=0, inference_time_ms=0,
            confusion_matrix=[[0,0,0],[0,0,0],[0,0,0]],
            classification_report={},
            optimal_thresholds={"high": 0, "medium": 0}
        )
    load_time = (time.time() - load_start) * 1000
    print(f"✅ Loaded in {load_time:.0f}ms")
    
    # Compute embeddings
    print(f"Computing embeddings for {len(texts)} texts...")
    inference_start = time.time()
    
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    # Compute similarities
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    scores = np.clip(similarities, 0, 1).tolist()
    
    inference_time = (time.time() - inference_start) * 1000
    print(f"✅ Inference completed in {inference_time:.0f}ms ({inference_time/len(texts):.1f}ms/text)")
    
    # Find optimal thresholds
    print("Finding optimal thresholds...")
    optimal_thresholds = find_optimal_thresholds(scores, y_true)
    print(f"Optimal thresholds: high >= {optimal_thresholds['high']:.2f}, medium >= {optimal_thresholds['medium']:.2f}")
    
    # Compute predictions with optimal thresholds
    y_pred = [score_to_label(s, optimal_thresholds) for s in scores]
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, scores)
    
    # Print summary
    print(f"\nResults with optimal thresholds:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Spearman ρ: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f})")
    print(f"  Precision@5: {metrics['precision_at_5']:.1%}")
    print(f"  Precision@10: {metrics['precision_at_10']:.1%}")
    print(f"  Precision@20: {metrics['precision_at_20']:.1%}")
    
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"           low  med  high")
    labels = ["low", "med", "high"]
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {labels[i]:>4}  {row}")
    
    return ModelResult(
        name=model_config["name"],
        description=model_config["description"],
        accuracy=metrics["accuracy"],
        spearman_rho=metrics["spearman_rho"],
        spearman_p=metrics["spearman_p"],
        precision_at_5=metrics["precision_at_5"],
        precision_at_10=metrics["precision_at_10"],
        precision_at_20=metrics["precision_at_20"],
        load_time_ms=load_time,
        inference_time_ms=inference_time,
        confusion_matrix=metrics["confusion_matrix"],
        classification_report=metrics["classification_report"],
        optimal_thresholds=optimal_thresholds,
    )


def run_comparison(data_path: str = "evaluation/labeled_risks.json") -> Dict:
    """Run full model comparison."""
    print("\n" + "="*70)
    print("EMBEDDING MODEL COMPARISON FOR RISK SCORING")
    print("="*70)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    y_true = [s["label"] for s in samples]
    
    print(f"\nDataset: {len(samples)} labeled samples")
    print(f"Distribution: {data['summary']['distribution']}")
    
    # Evaluate each model
    results = []
    for model_config in MODELS_TO_TEST:
        result = evaluate_model(model_config, texts, y_true)
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Header
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Spearman':<10} {'P@5':<8} {'P@10':<8} {'Time(ms)':<10}")
    print("-" * 71)
    
    # Results
    best_acc = max(r.accuracy for r in results)
    best_rho = max(r.spearman_rho for r in results)
    
    for r in results:
        acc_marker = "⭐" if r.accuracy == best_acc else "  "
        rho_marker = "⭐" if r.spearman_rho == best_rho else "  "
        
        print(f"{r.name:<25} {r.accuracy:>6.1%} {acc_marker} {r.spearman_rho:>6.3f} {rho_marker} "
              f"{r.precision_at_5:>6.0%}  {r.precision_at_10:>6.0%}  {r.inference_time_ms:>8.0f}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Score models (weighted combination)
    def model_score(r: ModelResult) -> float:
        return (r.accuracy * 0.3 + 
                r.spearman_rho * 0.3 + 
                r.precision_at_10 * 0.3 +
                (1 - min(r.inference_time_ms / 10000, 1)) * 0.1)  # Penalize slow models
    
    ranked = sorted(results, key=model_score, reverse=True)
    winner = ranked[0]
    
    print(f"\n🏆 Recommended model: {winner.name}")
    print(f"   Accuracy: {winner.accuracy:.1%}")
    print(f"   Spearman ρ: {winner.spearman_rho:.3f}")
    print(f"   Precision@10: {winner.precision_at_10:.1%}")
    print(f"   Optimal thresholds: high >= {winner.optimal_thresholds['high']:.2f}, "
          f"medium >= {winner.optimal_thresholds['medium']:.2f}")
    
    # Build output
    output = {
        "dataset": {
            "total_samples": len(samples),
            "distribution": data["summary"]["distribution"],
        },
        "models": [
            {
                "name": r.name,
                "description": r.description,
                "accuracy": r.accuracy,
                "spearman_rho": r.spearman_rho,
                "spearman_p": r.spearman_p,
                "precision_at_5": r.precision_at_5,
                "precision_at_10": r.precision_at_10,
                "precision_at_20": r.precision_at_20,
                "load_time_ms": r.load_time_ms,
                "inference_time_ms": r.inference_time_ms,
                "optimal_thresholds": r.optimal_thresholds,
                "confusion_matrix": r.confusion_matrix,
            }
            for r in results
        ],
        "recommendation": {
            "model": winner.name,
            "reason": f"Best overall performance with {winner.accuracy:.1%} accuracy and {winner.spearman_rho:.3f} Spearman correlation",
            "optimal_thresholds": winner.optimal_thresholds,
        }
    }
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding models for risk scoring")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/model_comparison_results.json", help="Output path")
    args = parser.parse_args()
    
    results = run_comparison(args.data)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

