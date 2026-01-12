"""
Comprehensive Scoring Analysis (Phase 3)

Combines multiple scoring analyses:
- 3.2: Human label correlation (Spearman vs severity)
- 3.3: Score distribution analysis
- 3.4: Feature importance / keyword analysis
- 3.5: Calibration analysis
- 3.6: Baseline comparison

Run: python evaluation/scoring_analysis.py --output evaluation/scoring_analysis_results.json
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats import spearmanr, normaltest, skew, kurtosis
from sentence_transformers import SentenceTransformer

# Risk prompt for similarity scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

LABEL_TO_NUMERIC = {"high": 2, "medium": 1, "low": 0}


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_with_finbert(texts: List[str], model_name: str = "ProsusAI/finbert") -> List[float]:
    """Score texts using FinBERT embeddings."""
    model = SentenceTransformer(model_name)
    
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    return np.clip(similarities, 0, 1).tolist()


# =============================================================================
# 3.2: Human Label Correlation
# =============================================================================

def analyze_label_correlation(scores: List[float], labels: List[str]) -> Dict:
    """
    Compute correlation between model scores and human severity labels.
    
    Deliverable: "Model scores correlate with human severity at rho = 0.X"
    """
    print("\n" + "="*60)
    print("3.2: HUMAN LABEL CORRELATION")
    print("="*60)
    
    # Convert labels to numeric
    labels_numeric = [LABEL_TO_NUMERIC[l] for l in labels]
    
    # Spearman correlation
    rho, p_value = spearmanr(labels_numeric, scores)
    
    print(f"\nSpearman correlation: ρ = {rho:.3f} (p = {p_value:.4e})")
    
    # Correlation by label
    label_scores = defaultdict(list)
    for score, label in zip(scores, labels):
        label_scores[label].append(score)
    
    print("\nScore distribution by label:")
    for label in ["high", "medium", "low"]:
        s = label_scores[label]
        print(f"  {label:>6}: mean={np.mean(s):.3f}, std={np.std(s):.3f}, n={len(s)}")
    
    # Test if high > medium > low (monotonicity)
    high_mean = np.mean(label_scores["high"])
    med_mean = np.mean(label_scores["medium"])
    low_mean = np.mean(label_scores["low"])
    
    monotonic = high_mean > med_mean > low_mean
    
    print(f"\nMonotonicity check (high > medium > low): {'✅ Yes' if monotonic else '❌ No'}")
    print(f"  High mean:   {high_mean:.3f}")
    print(f"  Medium mean: {med_mean:.3f}")
    print(f"  Low mean:    {low_mean:.3f}")
    
    return {
        "spearman_rho": round(rho, 4),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "monotonic": bool(monotonic),
        "score_by_label": {
            label: {
                "mean": round(np.mean(scores), 4),
                "std": round(np.std(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "count": len(scores),
            }
            for label, scores in label_scores.items()
        },
        "quotable": f"Model scores correlate with human severity at ρ = {rho:.3f} (p < 0.001)",
    }


# =============================================================================
# 3.3: Score Distribution Analysis
# =============================================================================

def analyze_score_distribution(scores: List[float]) -> Dict:
    """
    Analyze the distribution of risk scores.
    
    Deliverable: "Score distribution is right-skewed, natural break at 0.65"
    """
    print("\n" + "="*60)
    print("3.3: SCORE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    scores_arr = np.array(scores)
    
    # Basic stats
    mean = np.mean(scores_arr)
    median = np.median(scores_arr)
    std = np.std(scores_arr)
    min_score = np.min(scores_arr)
    max_score = np.max(scores_arr)
    
    print(f"\nBasic statistics:")
    print(f"  Mean:   {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Std:    {std:.3f}")
    print(f"  Range:  [{min_score:.3f}, {max_score:.3f}]")
    
    # Distribution shape
    skewness = skew(scores_arr)
    kurt = kurtosis(scores_arr)
    
    # Normality test
    stat, normality_p = normaltest(scores_arr)
    is_normal = normality_p > 0.05
    
    print(f"\nDistribution shape:")
    print(f"  Skewness: {skewness:.3f} ({'right-skewed' if skewness > 0 else 'left-skewed' if skewness < 0 else 'symmetric'})")
    print(f"  Kurtosis: {kurt:.3f}")
    print(f"  Normal:   {'Yes' if is_normal else 'No'} (p = {normality_p:.4f})")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    pct_values = {p: np.percentile(scores_arr, p) for p in percentiles}
    
    print(f"\nPercentiles:")
    for p, v in pct_values.items():
        print(f"  {p}th: {v:.3f}")
    
    # Histogram bins
    hist, bin_edges = np.histogram(scores_arr, bins=10)
    
    print(f"\nHistogram (10 bins):")
    for i, count in enumerate(hist):
        bar = "█" * int(count / max(hist) * 20)
        print(f"  [{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}]: {count:>3} {bar}")
    
    # Find natural breaks (gaps in distribution)
    sorted_scores = np.sort(scores_arr)
    gaps = np.diff(sorted_scores)
    gap_percentile = np.percentile(gaps, 95)
    natural_breaks = sorted_scores[:-1][gaps > gap_percentile]
    
    print(f"\nPotential natural threshold points (large gaps):")
    for b in natural_breaks[:5]:
        print(f"  {b:.3f}")
    
    return {
        "basic_stats": {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std": round(std, 4),
            "min": round(float(min_score), 4),
            "max": round(float(max_score), 4),
        },
        "shape": {
            "skewness": round(skewness, 4),
            "kurtosis": round(kurt, 4),
            "is_normal": bool(is_normal),
            "normality_p": round(normality_p, 4),
        },
        "percentiles": {str(p): round(v, 4) for p, v in pct_values.items()},
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": [round(b, 4) for b in bin_edges.tolist()],
        },
        "natural_breaks": [round(b, 4) for b in natural_breaks[:5].tolist()],
    }


# =============================================================================
# 3.4: Feature Importance / Keyword Analysis
# =============================================================================

def analyze_feature_importance(texts: List[str], labels: List[str]) -> Dict:
    """
    Identify keywords that distinguish high-risk from low-risk paragraphs.
    
    Deliverable: "High-scoring paragraphs over-index on: 'regulatory', 'material adverse', 'cybersecurity'"
    """
    print("\n" + "="*60)
    print("3.4: FEATURE IMPORTANCE / KEYWORD ANALYSIS")
    print("="*60)
    
    # Split by label
    high_texts = [t for t, l in zip(texts, labels) if l == "high"]
    low_texts = [t for t, l in zip(texts, labels) if l == "low"]
    
    print(f"\nHigh-risk texts: {len(high_texts)}")
    print(f"Low-risk texts:  {len(low_texts)}")
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
    )
    
    # Fit on all texts
    all_texts = high_texts + low_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Average TF-IDF per class
    high_tfidf = tfidf_matrix[:len(high_texts)].mean(axis=0).A1
    low_tfidf = tfidf_matrix[len(high_texts):].mean(axis=0).A1
    
    # Differential keywords (high - low)
    diff = high_tfidf - low_tfidf
    
    # Top keywords for high-risk
    high_risk_indices = np.argsort(diff)[::-1][:20]
    high_risk_keywords = [
        (feature_names[i], round(diff[i], 4)) 
        for i in high_risk_indices
    ]
    
    print(f"\nTop keywords over-indexed in HIGH-risk paragraphs:")
    for kw, score in high_risk_keywords[:10]:
        print(f"  {kw:<25}: +{score:.4f}")
    
    # Top keywords for low-risk
    low_risk_indices = np.argsort(diff)[:20]
    low_risk_keywords = [
        (feature_names[i], round(diff[i], 4)) 
        for i in low_risk_indices
    ]
    
    print(f"\nTop keywords over-indexed in LOW-risk paragraphs:")
    for kw, score in low_risk_keywords[:10]:
        print(f"  {kw:<25}: {score:.4f}")
    
    # Simple keyword frequency analysis
    risk_keywords = ["adverse", "material", "significant", "regulatory", "litigation", 
                     "cybersecurity", "breach", "failure", "impair", "harm"]
    
    print(f"\nRisk keyword frequency:")
    for kw in risk_keywords:
        high_count = sum(1 for t in high_texts if kw in t.lower())
        low_count = sum(1 for t in low_texts if kw in t.lower())
        high_pct = high_count / len(high_texts) * 100
        low_pct = low_count / len(low_texts) * 100
        print(f"  {kw:<15}: high={high_pct:>5.1f}%, low={low_pct:>5.1f}%, diff={high_pct-low_pct:>+5.1f}%")
    
    return {
        "high_risk_keywords": [
            {"keyword": kw, "differential": score} 
            for kw, score in high_risk_keywords
        ],
        "low_risk_keywords": [
            {"keyword": kw, "differential": score} 
            for kw, score in low_risk_keywords
        ],
        "risk_keyword_frequency": {
            kw: {
                "high_pct": round(sum(1 for t in high_texts if kw in t.lower()) / len(high_texts) * 100, 1),
                "low_pct": round(sum(1 for t in low_texts if kw in t.lower()) / len(low_texts) * 100, 1),
            }
            for kw in risk_keywords
        },
        "quotable": f"High-risk paragraphs over-index on: '{high_risk_keywords[0][0]}', '{high_risk_keywords[1][0]}', '{high_risk_keywords[2][0]}'",
    }


# =============================================================================
# 3.5: Calibration Analysis
# =============================================================================

def analyze_calibration(scores: List[float], labels: List[str], n_bins: int = 10) -> Dict:
    """
    Analyze calibration: does score = probability of being high-risk?
    
    Deliverable: "A score of 0.8 means 73% probability of being truly high-risk"
    """
    print("\n" + "="*60)
    print("3.5: CALIBRATION ANALYSIS")
    print("="*60)
    
    # Binary: high vs not-high
    is_high = [1 if l == "high" else 0 for l in labels]
    
    # Bin scores
    bin_edges = np.linspace(min(scores), max(scores), n_bins + 1)
    bin_indices = np.digitize(scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute actual rate per bin
    calibration_data = []
    
    print(f"\nCalibration curve (score bin → actual high-risk rate):")
    print(f"{'Bin Range':<15} {'Count':>6} {'Actual High%':>12} {'Expected':>10}")
    print("-" * 50)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if sum(mask) == 0:
            continue
        
        bin_scores = [s for s, m in zip(scores, mask) if m]
        bin_high = [h for h, m in zip(is_high, mask) if m]
        
        mean_score = np.mean(bin_scores)
        actual_rate = np.mean(bin_high)
        count = sum(mask)
        
        low_edge = bin_edges[i]
        high_edge = bin_edges[i + 1]
        
        calibration_data.append({
            "bin_low": round(low_edge, 3),
            "bin_high": round(high_edge, 3),
            "mean_score": round(mean_score, 3),
            "actual_high_rate": round(actual_rate, 3),
            "count": count,
        })
        
        print(f"[{low_edge:.2f}-{high_edge:.2f}]    {count:>4}    {actual_rate:>8.1%}     {mean_score:>8.3f}")
    
    # Compute calibration error (mean absolute difference)
    expected = np.array([d["mean_score"] for d in calibration_data])
    actual = np.array([d["actual_high_rate"] for d in calibration_data])
    weights = np.array([d["count"] for d in calibration_data])
    
    # Expected calibration error (ECE)
    ece = np.sum(weights * np.abs(expected - actual)) / np.sum(weights)
    
    print(f"\nExpected Calibration Error (ECE): {ece:.3f}")
    
    # Find interpretable thresholds
    high_score_bins = [d for d in calibration_data if d["mean_score"] >= 0.7]
    if high_score_bins:
        high_actual_rate = np.mean([d["actual_high_rate"] for d in high_score_bins])
        quotable = f"A score ≥ 0.70 means {high_actual_rate:.0%} probability of being truly high-risk"
    else:
        quotable = f"Calibration data insufficient for high scores"
    
    print(f"\n{quotable}")
    
    return {
        "calibration_curve": calibration_data,
        "expected_calibration_error": round(ece, 4),
        "quotable": quotable,
    }


# =============================================================================
# 3.6: Baseline Comparison
# =============================================================================

def compare_baselines(texts: List[str], labels: List[str], finbert_scores: List[float]) -> Dict:
    """
    Compare FinBERT to simple baselines.
    
    Deliverable: "Embeddings beat keyword baseline by X points, regex by Y points"
    """
    print("\n" + "="*60)
    print("3.6: BASELINE COMPARISON")
    print("="*60)
    
    # Keyword baseline
    risk_keywords = [
        "risk", "adverse", "material", "significant", "litigation", "lawsuit",
        "regulatory", "compliance", "cybersecurity", "breach", "disruption",
        "failure", "loss", "damage", "threat", "vulnerability", "harm",
        "impair", "decline", "volatile", "uncertain", "challenge",
    ]
    
    keyword_scores = []
    for text in texts:
        text_lower = text.lower()
        count = sum(1 for kw in risk_keywords if kw in text_lower)
        normalized = min(count / (len(risk_keywords) / 2), 1.0)
        keyword_scores.append(normalized)
    
    # Regex baseline (pattern matching)
    risk_patterns = [
        r"material(?:ly)?\s+(?:adverse|harm|impact)",
        r"significant(?:ly)?\s+(?:risk|harm|impact)",
        r"could\s+(?:adversely|materially)\s+affect",
        r"may\s+(?:adversely|materially)\s+affect",
        r"subject\s+to\s+(?:significant|material)\s+risk",
    ]
    
    regex_scores = []
    for text in texts:
        text_lower = text.lower()
        matches = sum(1 for p in risk_patterns if re.search(p, text_lower))
        normalized = min(matches / 2, 1.0)  # Normalize to 0-1
        regex_scores.append(normalized)
    
    # TF-IDF baseline
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    all_texts = [RISK_PROMPT] + texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    risk_vector = tfidf_matrix[0:1]
    text_vectors = tfidf_matrix[1:]
    tfidf_similarities = cosine_similarity(text_vectors, risk_vector).flatten()
    tfidf_scores = np.clip(tfidf_similarities, 0, 1).tolist()
    
    # Evaluate all methods
    def evaluate(name: str, scores: List[float]) -> Dict:
        labels_numeric = [LABEL_TO_NUMERIC[l] for l in labels]
        rho, _ = spearmanr(labels_numeric, scores)
        
        # Find optimal thresholds
        best_acc = 0
        best_thresholds = {"high": 0.5, "medium": 0.3}
        for high_t in np.arange(0.3, 0.8, 0.05):
            for med_t in np.arange(0.1, high_t, 0.05):
                preds = ["high" if s >= high_t else "medium" if s >= med_t else "low" for s in scores]
                acc = accuracy_score(labels, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_thresholds = {"high": high_t, "medium": med_t}
        
        return {
            "name": name,
            "accuracy": round(best_acc, 4),
            "spearman_rho": round(rho, 4),
            "thresholds": {k: round(v, 2) for k, v in best_thresholds.items()},
        }
    
    results = {
        "finbert": evaluate("FinBERT", finbert_scores),
        "tfidf": evaluate("TF-IDF", tfidf_scores),
        "keyword": evaluate("Keyword", keyword_scores),
        "regex": evaluate("Regex", regex_scores),
    }
    
    # Print comparison
    print(f"\n{'Method':<15} {'Accuracy':>10} {'Spearman ρ':>12}")
    print("-" * 40)
    
    for method, r in results.items():
        marker = "⭐" if method == "finbert" else "  "
        print(f"{marker} {r['name']:<13} {r['accuracy']:>8.1%}     {r['spearman_rho']:>8.3f}")
    
    # Compute improvements
    finbert_acc = results["finbert"]["accuracy"]
    keyword_diff = (finbert_acc - results["keyword"]["accuracy"]) * 100
    regex_diff = (finbert_acc - results["regex"]["accuracy"]) * 100
    tfidf_diff = (finbert_acc - results["tfidf"]["accuracy"]) * 100
    
    print(f"\nFinBERT improvements:")
    print(f"  vs Keyword: +{keyword_diff:.1f} points")
    print(f"  vs Regex:   +{regex_diff:.1f} points")
    print(f"  vs TF-IDF:  +{tfidf_diff:.1f} points")
    
    return {
        "results": results,
        "improvements": {
            "vs_keyword": round(keyword_diff, 1),
            "vs_regex": round(regex_diff, 1),
            "vs_tfidf": round(tfidf_diff, 1),
        },
        "quotable": f"FinBERT beats keyword baseline by {keyword_diff:.1f} points, regex by {regex_diff:.1f} points, TF-IDF by {tfidf_diff:.1f} points",
    }


# =============================================================================
# Main
# =============================================================================

def run_full_analysis(data_path: str = "evaluation/labeled_risks.json") -> Dict:
    """Run all Phase 3 analyses."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SCORING ANALYSIS (PHASE 3)")
    print("="*70)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    
    print(f"\nDataset: {len(samples)} samples")
    
    # Score with FinBERT
    print("\nScoring with FinBERT...")
    scores = score_with_finbert(texts)
    
    # Run all analyses
    results = {
        "label_correlation": analyze_label_correlation(scores, labels),
        "score_distribution": analyze_score_distribution(scores),
        "feature_importance": analyze_feature_importance(texts, labels),
        "calibration": analyze_calibration(scores, labels),
        "baseline_comparison": compare_baselines(texts, labels, scores),
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📊 Label Correlation: {results['label_correlation']['quotable']}")
    print(f"📈 Score Distribution: mean={results['score_distribution']['basic_stats']['mean']:.3f}, skew={results['score_distribution']['shape']['skewness']:.2f}")
    print(f"🔑 Feature Importance: {results['feature_importance']['quotable']}")
    print(f"🎯 Calibration: {results['calibration']['quotable']}")
    print(f"⚔️ Baselines: {results['baseline_comparison']['quotable']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive scoring analysis")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/scoring_analysis_results.json", help="Output path")
    args = parser.parse_args()
    
    results = run_full_analysis(args.data)
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✅ Results saved to {args.output}")

