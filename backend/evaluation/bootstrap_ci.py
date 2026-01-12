"""
Bootstrap Confidence Intervals for Risk Scoring Evaluation.

Implements bootstrap resampling to compute 95% confidence intervals
for all evaluation metrics: accuracy, Spearman ρ, Precision@k.

Deliverable: "Accuracy: 54.5% (95% CI: 48.2% - 60.8%)"
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
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


@dataclass
class MetricWithCI:
    """A metric value with its 95% confidence interval."""
    value: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    
    def __str__(self) -> str:
        return f"{self.value:.1%} (95% CI: {self.ci_lower:.1%} - {self.ci_upper:.1%})"
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
        }


@dataclass 
class BootstrapResults:
    """Complete bootstrap evaluation results."""
    n_samples: int
    n_bootstrap: int
    model_name: str
    accuracy: MetricWithCI
    spearman_rho: MetricWithCI
    precision_at_5: MetricWithCI
    precision_at_10: MetricWithCI
    precision_at_20: MetricWithCI
    thresholds: Dict[str, float]


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_to_label(score: float, thresholds: Dict[str, float]) -> str:
    """Convert numeric score to label category."""
    if score >= thresholds["high"]:
        return "high"
    elif score >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def compute_metrics_single(y_true: List[str], y_pred: List[str], 
                           scores: List[float]) -> Dict[str, float]:
    """Compute metrics for a single bootstrap sample."""
    y_true_num = [LABEL_TO_NUMERIC[l] for l in y_true]
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Spearman correlation
    if len(set(y_true_num)) > 1 and len(set(scores)) > 1:
        spearman_corr, _ = spearmanr(y_true_num, scores)
    else:
        spearman_corr = 0.0
    
    # Precision@k
    sorted_indices = np.argsort(scores)[::-1]
    
    p_at_5 = sum(1 for i in sorted_indices[:5] if y_true[i] == "high") / 5
    p_at_10 = sum(1 for i in sorted_indices[:10] if y_true[i] == "high") / 10
    p_at_20 = sum(1 for i in sorted_indices[:20] if y_true[i] == "high") / 20
    
    return {
        "accuracy": accuracy,
        "spearman_rho": spearman_corr,
        "precision_at_5": p_at_5,
        "precision_at_10": p_at_10,
        "precision_at_20": p_at_20,
    }


def bootstrap_confidence_interval(values: np.ndarray, 
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval from bootstrap samples.
    
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    point_estimate = np.mean(values)
    ci_lower = np.percentile(values, lower_percentile)
    ci_upper = np.percentile(values, upper_percentile)
    
    return point_estimate, ci_lower, ci_upper


def run_bootstrap_evaluation(
    model_name: str = "ProsusAI/finbert",
    thresholds: Dict[str, float] = None,
    n_bootstrap: int = 1000,
    data_path: str = "evaluation/labeled_risks.json",
    seed: int = 42,
) -> BootstrapResults:
    """
    Run bootstrap evaluation to compute metrics with confidence intervals.
    
    Args:
        model_name: Sentence transformer model to use
        thresholds: Score thresholds for classification
        n_bootstrap: Number of bootstrap iterations
        data_path: Path to labeled data
        seed: Random seed for reproducibility
        
    Returns:
        BootstrapResults with all metrics and CIs
    """
    np.random.seed(seed)
    
    if thresholds is None:
        # Use optimal thresholds from model comparison
        thresholds = {"high": 0.70, "medium": 0.65}
    
    print("\n" + "="*70)
    print("BOOTSTRAP CONFIDENCE INTERVAL EVALUATION")
    print("="*70)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    y_true = [s["label"] for s in samples]
    n = len(samples)
    
    print(f"\nDataset: {n} labeled samples")
    print(f"Distribution: {data['summary']['distribution']}")
    print(f"Model: {model_name}")
    print(f"Thresholds: high >= {thresholds['high']}, medium >= {thresholds['medium']}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    
    # Load model and compute embeddings once
    print(f"\nLoading model and computing embeddings...")
    model = SentenceTransformer(model_name)
    
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    # Compute all scores
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    all_scores = np.clip(similarities, 0, 1)
    all_preds = [score_to_label(s, thresholds) for s in all_scores]
    
    # Point estimates on full data
    print("\nComputing point estimates...")
    full_metrics = compute_metrics_single(y_true, all_preds, all_scores.tolist())
    
    print(f"\nPoint estimates (full dataset):")
    print(f"  Accuracy: {full_metrics['accuracy']:.1%}")
    print(f"  Spearman ρ: {full_metrics['spearman_rho']:.3f}")
    print(f"  Precision@5: {full_metrics['precision_at_5']:.0%}")
    print(f"  Precision@10: {full_metrics['precision_at_10']:.0%}")
    print(f"  Precision@20: {full_metrics['precision_at_20']:.0%}")
    
    # Bootstrap resampling
    print(f"\nRunning {n_bootstrap} bootstrap iterations...")
    
    bootstrap_metrics = {
        "accuracy": [],
        "spearman_rho": [],
        "precision_at_5": [],
        "precision_at_10": [],
        "precision_at_20": [],
    }
    
    start_time = time.time()
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}...")
        
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        # Get resampled data
        y_true_boot = [y_true[j] for j in indices]
        y_pred_boot = [all_preds[j] for j in indices]
        scores_boot = [all_scores[j] for j in indices]
        
        # Compute metrics
        metrics = compute_metrics_single(y_true_boot, y_pred_boot, scores_boot)
        
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(metrics[key])
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({elapsed/n_bootstrap*1000:.1f}ms/iteration)")
    
    # Compute confidence intervals
    print("\nComputing 95% confidence intervals...")
    
    results = {}
    for metric_name, values in bootstrap_metrics.items():
        values_arr = np.array(values)
        point, lower, upper = bootstrap_confidence_interval(values_arr)
        
        results[metric_name] = MetricWithCI(
            value=full_metrics[metric_name],  # Use actual point estimate, not bootstrap mean
            ci_lower=lower,
            ci_upper=upper,
            ci_width=upper - lower,
        )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Value':<12} {'95% CI':<25} {'Width':<10}")
    print("-" * 67)
    
    metric_display = [
        ("Accuracy", results["accuracy"], True),
        ("Spearman ρ", results["spearman_rho"], False),
        ("Precision@5", results["precision_at_5"], True),
        ("Precision@10", results["precision_at_10"], True),
        ("Precision@20", results["precision_at_20"], True),
    ]
    
    for name, m, is_pct in metric_display:
        if is_pct:
            print(f"{name:<20} {m.value:>10.1%}  ({m.ci_lower:.1%} - {m.ci_upper:.1%})  {m.ci_width:>8.1%}")
        else:
            print(f"{name:<20} {m.value:>10.3f}  ({m.ci_lower:.3f} - {m.ci_upper:.3f})  {m.ci_width:>8.3f}")
    
    # Key deliverable format
    print("\n" + "="*70)
    print("DELIVERABLE FORMATS")
    print("="*70)
    
    acc = results["accuracy"]
    rho = results["spearman_rho"]
    p10 = results["precision_at_10"]
    
    print(f"\nAccuracy: {acc.value:.1%} (95% CI: {acc.ci_lower:.1%} - {acc.ci_upper:.1%})")
    print(f"Spearman ρ: {rho.value:.3f} (95% CI: {rho.ci_lower:.3f} - {rho.ci_upper:.3f})")
    print(f"Precision@10: {p10.value:.0%} (95% CI: {p10.ci_lower:.0%} - {p10.ci_upper:.0%})")
    
    return BootstrapResults(
        n_samples=n,
        n_bootstrap=n_bootstrap,
        model_name=model_name,
        accuracy=results["accuracy"],
        spearman_rho=results["spearman_rho"],
        precision_at_5=results["precision_at_5"],
        precision_at_10=results["precision_at_10"],
        precision_at_20=results["precision_at_20"],
        thresholds=thresholds,
    )


def results_to_dict(results: BootstrapResults) -> Dict:
    """Convert results to JSON-serializable dict."""
    return {
        "n_samples": results.n_samples,
        "n_bootstrap": results.n_bootstrap,
        "model_name": results.model_name,
        "thresholds": results.thresholds,
        "metrics": {
            "accuracy": results.accuracy.to_dict(),
            "spearman_rho": results.spearman_rho.to_dict(),
            "precision_at_5": results.precision_at_5.to_dict(),
            "precision_at_10": results.precision_at_10.to_dict(),
            "precision_at_20": results.precision_at_20.to_dict(),
        },
        "summary": {
            "accuracy": f"{results.accuracy.value:.1%} (95% CI: {results.accuracy.ci_lower:.1%} - {results.accuracy.ci_upper:.1%})",
            "spearman_rho": f"{results.spearman_rho.value:.3f} (95% CI: {results.spearman_rho.ci_lower:.3f} - {results.spearman_rho.ci_upper:.3f})",
            "precision_at_10": f"{results.precision_at_10.value:.0%} (95% CI: {results.precision_at_10.ci_lower:.0%} - {results.precision_at_10.ci_upper:.0%})",
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for evaluation")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--model", default="ProsusAI/finbert", help="Model to evaluate")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--high-threshold", type=float, default=0.70, help="Threshold for high risk")
    parser.add_argument("--medium-threshold", type=float, default=0.65, help="Threshold for medium risk")
    parser.add_argument("--output", default="evaluation/bootstrap_results.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    thresholds = {"high": args.high_threshold, "medium": args.medium_threshold}
    
    results = run_bootstrap_evaluation(
        model_name=args.model,
        thresholds=thresholds,
        n_bootstrap=args.n_bootstrap,
        data_path=args.data,
        seed=args.seed,
    )
    
    # Save results
    output = results_to_dict(results)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

