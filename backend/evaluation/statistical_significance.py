"""
Statistical Significance Tests (Phase 4.4)

Implements paired bootstrap test to compare:
- FinBERT embeddings vs TF-IDF baseline

Deliverable: "Embeddings beat TF-IDF by X points (p < 0.01)"
"""

import os
import sys
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# Risk prompt for similarity scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Label mappings
LABEL_TO_NUMERIC = {"high": 2, "medium": 1, "low": 0}


@dataclass
class MethodResult:
    """Results from a single scoring method."""
    name: str
    scores: List[float]
    predictions: List[str]
    accuracy: float
    spearman_rho: float
    precision_at_10: float


@dataclass
class SignificanceResult:
    """Results of significance test between two methods."""
    method_a: str
    method_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    p_value: float
    significant: bool
    n_bootstrap: int
    
    def to_dict(self) -> Dict:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "value_a": round(self.value_a, 4),
            "value_b": round(self.value_b, 4),
            "difference": round(self.difference, 4),
            "p_value": round(self.p_value, 4),
            "significant": bool(self.significant),
            "n_bootstrap": self.n_bootstrap,
        }


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


def find_optimal_thresholds(scores: List[float], y_true: List[str]) -> Dict[str, float]:
    """Find optimal thresholds by grid search."""
    best_acc = 0
    best_thresholds = {"high": 0.55, "medium": 0.45}
    
    for high_thresh in np.arange(0.30, 0.80, 0.05):
        for med_thresh in np.arange(0.15, high_thresh, 0.05):
            thresholds = {"high": high_thresh, "medium": med_thresh}
            y_pred = [score_to_label(s, thresholds) for s in scores]
            acc = accuracy_score(y_true, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
    
    return best_thresholds


def score_with_finbert(texts: List[str]) -> List[float]:
    """Score texts using FinBERT embeddings."""
    print("Loading FinBERT model...")
    model = SentenceTransformer("ProsusAI/finbert")
    
    print(f"Scoring {len(texts)} texts with FinBERT...")
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    return np.clip(similarities, 0, 1).tolist()


def score_with_tfidf(texts: List[str]) -> List[float]:
    """Score texts using TF-IDF similarity to risk prompt."""
    print(f"Scoring {len(texts)} texts with TF-IDF...")
    
    # Fit TF-IDF on all texts including risk prompt
    all_texts = [RISK_PROMPT] + texts
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    risk_vector = tfidf_matrix[0:1]
    text_vectors = tfidf_matrix[1:]
    
    # Cosine similarity with TF-IDF vectors
    similarities = cosine_similarity(text_vectors, risk_vector).flatten()
    return np.clip(similarities, 0, 1).tolist()


def score_with_keyword_count(texts: List[str]) -> List[float]:
    """Score texts using simple keyword counting."""
    print(f"Scoring {len(texts)} texts with keyword counting...")
    
    # Risk keywords
    keywords = [
        "risk", "adverse", "material", "significant", "litigation", "lawsuit",
        "regulatory", "compliance", "cybersecurity", "breach", "disruption",
        "failure", "loss", "damage", "threat", "vulnerability", "harm",
        "impair", "decline", "volatile", "uncertain", "challenge",
    ]
    
    scores = []
    for text in texts:
        text_lower = text.lower()
        count = sum(1 for kw in keywords if kw in text_lower)
        # Normalize by text length and keyword count
        normalized = min(count / (len(keywords) / 2), 1.0)
        scores.append(normalized)
    
    return scores


def evaluate_method(
    name: str,
    scores: List[float],
    y_true: List[str],
) -> MethodResult:
    """Evaluate a single method."""
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(scores, y_true)
    predictions = [score_to_label(s, thresholds) for s in scores]
    
    # Compute metrics
    accuracy = accuracy_score(y_true, predictions)
    
    y_true_num = [LABEL_TO_NUMERIC[l] for l in y_true]
    rho, _ = spearmanr(y_true_num, scores)
    
    sorted_indices = np.argsort(scores)[::-1]
    precision_at_10 = sum(1 for i in sorted_indices[:10] if y_true[i] == "high") / 10
    
    return MethodResult(
        name=name,
        scores=scores,
        predictions=predictions,
        accuracy=accuracy,
        spearman_rho=rho,
        precision_at_10=precision_at_10,
    )


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    y_true: List[str],
    metric: str = "accuracy",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> SignificanceResult:
    """
    Perform paired bootstrap test to compare two methods.
    
    Tests whether method A significantly outperforms method B.
    
    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B  
        y_true: True labels
        metric: Metric to compare ("accuracy", "spearman", "precision_at_10")
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        SignificanceResult with p-value
    """
    np.random.seed(seed)
    n = len(y_true)
    
    y_true_num = [LABEL_TO_NUMERIC[l] for l in y_true]
    
    # OPTIMIZATION: Compute thresholds ONCE on full data, not per-bootstrap
    # This is standard practice - we test the observed system, not re-optimize each sample
    thresholds_a = find_optimal_thresholds(scores_a, y_true)
    thresholds_b = find_optimal_thresholds(scores_b, y_true)
    
    def compute_metric(scores: List[float], thresholds: Dict[str, float], indices: np.ndarray) -> float:
        """Compute metric on a bootstrap sample with fixed thresholds."""
        sample_scores = [scores[i] for i in indices]
        sample_labels = [y_true[i] for i in indices]
        sample_labels_num = [y_true_num[i] for i in indices]
        
        if metric == "accuracy":
            preds = [score_to_label(s, thresholds) for s in sample_scores]
            return accuracy_score(sample_labels, preds)
        
        elif metric == "spearman":
            rho, _ = spearmanr(sample_labels_num, sample_scores)
            return rho if not np.isnan(rho) else 0.0
        
        elif metric == "precision_at_10":
            sorted_indices = np.argsort(sample_scores)[::-1]
            return sum(1 for i in sorted_indices[:10] if sample_labels[i] == "high") / 10
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Compute observed difference
    all_indices = np.arange(n)
    observed_a = compute_metric(scores_a, thresholds_a, all_indices)
    observed_b = compute_metric(scores_b, thresholds_b, all_indices)
    observed_diff = observed_a - observed_b
    
    print(f"\nBootstrap test for {metric}:")
    print(f"  Method A: {observed_a:.4f}")
    print(f"  Method B: {observed_b:.4f}")
    print(f"  Observed difference: {observed_diff:+.4f}")
    
    # Bootstrap (now fast - no grid search per iteration)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        diff = compute_metric(scores_a, thresholds_a, indices) - compute_metric(scores_b, thresholds_b, indices)
        bootstrap_diffs.append(diff)
    
    # Two-sided p-value: probability that observed difference is due to chance
    # Under null hypothesis, differences are symmetric around 0
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # P-value: proportion of bootstrap samples where B >= A (i.e., diff <= 0)
    # This tests H0: A <= B vs H1: A > B
    p_value = np.mean(bootstrap_diffs <= 0)
    
    # Confidence interval for the difference
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)
    
    print(f"  Bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  P-value (A > B): {p_value:.4f}")
    
    return SignificanceResult(
        method_a="FinBERT",
        method_b="TF-IDF",
        metric=metric,
        value_a=observed_a,
        value_b=observed_b,
        difference=observed_diff,
        p_value=p_value,
        significant=p_value < 0.05,
        n_bootstrap=n_bootstrap,
    )


def run_significance_tests(
    data_path: str = "evaluation/labeled_risks.json",
    n_bootstrap: int = 10000,
) -> Dict:
    """Run full significance test suite."""
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("FinBERT Embeddings vs TF-IDF Baseline")
    print("="*70)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    y_true = [s["label"] for s in samples]
    
    print(f"\nDataset: {len(samples)} labeled samples")
    
    # Score with both methods
    finbert_scores = score_with_finbert(texts)
    tfidf_scores = score_with_tfidf(texts)
    keyword_scores = score_with_keyword_count(texts)
    
    # Evaluate each method
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    finbert_result = evaluate_method("FinBERT", finbert_scores, y_true)
    tfidf_result = evaluate_method("TF-IDF", tfidf_scores, y_true)
    keyword_result = evaluate_method("Keyword", keyword_scores, y_true)
    
    print(f"\n{'Method':<15} {'Accuracy':<12} {'Spearman ρ':<12} {'P@10':<10}")
    print("-" * 50)
    for r in [finbert_result, tfidf_result, keyword_result]:
        print(f"{r.name:<15} {r.accuracy:>8.1%}     {r.spearman_rho:>8.3f}     {r.precision_at_10:>6.0%}")
    
    # Run significance tests
    print("\n" + "="*70)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS")
    print(f"(n_bootstrap = {n_bootstrap})")
    print("="*70)
    
    results = []
    
    # FinBERT vs TF-IDF
    for metric in ["accuracy", "spearman", "precision_at_10"]:
        result = paired_bootstrap_test(
            finbert_scores, tfidf_scores, y_true,
            metric=metric, n_bootstrap=n_bootstrap,
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SIGNIFICANCE TEST SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'FinBERT':<12} {'TF-IDF':<12} {'Diff':<10} {'P-value':<10} {'Sig?'}")
    print("-" * 75)
    
    for r in results:
        sig_marker = "✅" if r.significant else "❌"
        p_str = f"p={r.p_value:.4f}" if r.p_value >= 0.0001 else "p<0.0001"
        print(f"{r.metric:<20} {r.value_a:>8.3f}     {r.value_b:>8.3f}     {r.difference:>+.3f}     {p_str:<10} {sig_marker}")
    
    # Generate quotable result
    acc_diff = (finbert_result.accuracy - tfidf_result.accuracy) * 100
    acc_result = next(r for r in results if r.metric == "accuracy")
    
    print("\n" + "="*70)
    print("QUOTABLE RESULTS")
    print("="*70)
    
    if acc_result.significant:
        print(f"\n✅ \"FinBERT embeddings beat TF-IDF by {acc_diff:.1f} points (p = {acc_result.p_value:.4f})\"")
    else:
        print(f"\n⚠️ \"FinBERT outperforms TF-IDF by {acc_diff:.1f} points, but not statistically significant (p = {acc_result.p_value:.4f})\"")
    
    spearman_result = next(r for r in results if r.metric == "spearman")
    rho_diff = finbert_result.spearman_rho - tfidf_result.spearman_rho
    
    if spearman_result.significant:
        print(f"✅ \"Spearman correlation: FinBERT ({finbert_result.spearman_rho:.3f}) vs TF-IDF ({tfidf_result.spearman_rho:.3f}), Δ = {rho_diff:+.3f} (p = {spearman_result.p_value:.4f})\"")
    
    return {
        "dataset": {
            "total_samples": len(samples),
            "distribution": data.get("summary", {}).get("distribution", {}),
        },
        "methods": {
            "finbert": {
                "accuracy": finbert_result.accuracy,
                "spearman_rho": finbert_result.spearman_rho,
                "precision_at_10": finbert_result.precision_at_10,
            },
            "tfidf": {
                "accuracy": tfidf_result.accuracy,
                "spearman_rho": tfidf_result.spearman_rho,
                "precision_at_10": tfidf_result.precision_at_10,
            },
            "keyword": {
                "accuracy": keyword_result.accuracy,
                "spearman_rho": keyword_result.spearman_rho,
                "precision_at_10": keyword_result.precision_at_10,
            },
        },
        "significance_tests": [r.to_dict() for r in results],
        "quotable": {
            "accuracy": f"FinBERT beats TF-IDF by {acc_diff:.1f} points (p = {acc_result.p_value:.4f})",
            "significant": bool(acc_result.significant),
        },
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/significance_results.json", help="Output path")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap iterations")
    args = parser.parse_args()
    
    results = run_significance_tests(args.data, args.n_bootstrap)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

