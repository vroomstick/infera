"""
Compare TF-IDF vs Embedding scoring methods.
Generates comparison metrics, plots, and analysis.
"""

import os
import sys
import json
from typing import Dict, List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

from evaluation.eval_scorer import (
    load_labeled_data, 
    score_with_tfidf, 
    score_with_embeddings,
    score_to_label,
    LABEL_TO_NUMERIC
)


def generate_plots(samples: List[Dict], tfidf_scores: List[float], 
                   emb_scores: List[float], output_dir: str = "evaluation/plots"):
    """Generate evaluation plots."""
    if not HAS_PLOTTING:
        print("Skipping plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = [s["label"] for s in samples]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Score Distribution by Label (Embeddings)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Embeddings distribution
    ax = axes[0]
    for label in ["low", "medium", "high"]:
        label_scores = [emb_scores[i] for i, l in enumerate(y_true) if l == label]
        ax.hist(label_scores, bins=15, alpha=0.6, label=f"{label} (n={len(label_scores)})")
    ax.set_xlabel("Embedding Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by True Label (Embeddings)")
    ax.legend()
    ax.axvline(x=0.55, color='r', linestyle='--', alpha=0.5, label='high threshold')
    ax.axvline(x=0.45, color='orange', linestyle='--', alpha=0.5, label='medium threshold')
    
    # TF-IDF distribution
    ax = axes[1]
    for label in ["low", "medium", "high"]:
        label_scores = [tfidf_scores[i] for i, l in enumerate(y_true) if l == label]
        ax.hist(label_scores, bins=15, alpha=0.6, label=f"{label} (n={len(label_scores)})")
    ax.set_xlabel("TF-IDF Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by True Label (TF-IDF)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/score_distribution.png")
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (method, scores) in enumerate([("TF-IDF", tfidf_scores), ("Embeddings", emb_scores)]):
        ax = axes[idx]
        y_pred = [score_to_label(s, "tfidf" if method == "TF-IDF" else "embedding") for s in scores]
        
        # Build confusion matrix
        labels = ["low", "medium", "high"]
        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[labels.index(true), labels.index(pred)] += 1
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {method}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/confusion_matrix.png")
    
    # 3. Method Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute metrics for both methods
    metrics = {
        "TF-IDF": {},
        "Embeddings": {}
    }
    
    for method, scores in [("TF-IDF", tfidf_scores), ("Embeddings", emb_scores)]:
        method_type = "tfidf" if method == "TF-IDF" else "embedding"
        y_pred = [score_to_label(s, method_type) for s in scores]
        
        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        metrics[method]["Accuracy"] = correct / len(y_true)
        
        # Precision@5
        sorted_indices = np.argsort(scores)[::-1]
        metrics[method]["Precision@5"] = sum(1 for i in sorted_indices[:5] if y_true[i] == "high") / 5
        
        # Precision@10
        metrics[method]["Precision@10"] = sum(1 for i in sorted_indices[:10] if y_true[i] == "high") / 10
    
    # Plot
    x = np.arange(3)
    width = 0.35
    metric_names = ["Accuracy", "Precision@5", "Precision@10"]
    
    tfidf_vals = [metrics["TF-IDF"][m] for m in metric_names]
    emb_vals = [metrics["Embeddings"][m] for m in metric_names]
    
    bars1 = ax.bar(x - width/2, tfidf_vals, width, label='TF-IDF', color='#ff7f0e')
    bars2 = ax.bar(x + width/2, emb_vals, width, label='Embeddings', color='#1f77b4')
    
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison: TF-IDF vs Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/method_comparison.png")
    
    # 4. Score Scatter Plot (TF-IDF vs Embeddings)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    for label in ["low", "medium", "high"]:
        mask = [l == label for l in y_true]
        x_vals = [tfidf_scores[i] for i, m in enumerate(mask) if m]
        y_vals = [emb_scores[i] for i, m in enumerate(mask) if m]
        ax.scatter(x_vals, y_vals, c=colors[label], label=label, alpha=0.7, s=60)
    
    ax.set_xlabel("TF-IDF Score")
    ax.set_ylabel("Embedding Score")
    ax.set_title("TF-IDF vs Embedding Scores (colored by true label)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/score_scatter.png")


def identify_errors(samples: List[Dict], tfidf_scores: List[float], 
                    emb_scores: List[float]) -> Dict:
    """Identify and categorize prediction errors."""
    y_true = [s["label"] for s in samples]
    emb_pred = [score_to_label(s, "embedding") for s in emb_scores]
    tfidf_pred = [score_to_label(s, "tfidf") for s in tfidf_scores]
    
    errors = {
        "embedding": {"false_positives": [], "false_negatives": []},
        "tfidf": {"false_positives": [], "false_negatives": []}
    }
    
    for i, sample in enumerate(samples):
        # Embedding errors
        if emb_pred[i] == "high" and y_true[i] == "low":
            errors["embedding"]["false_positives"].append({
                "id": sample["id"],
                "source": sample["source"],
                "true": y_true[i],
                "pred": emb_pred[i],
                "score": round(emb_scores[i], 4),
                "text": sample["text"][:200] + "...",
                "notes": sample.get("notes", "")
            })
        elif emb_pred[i] == "low" and y_true[i] == "high":
            errors["embedding"]["false_negatives"].append({
                "id": sample["id"],
                "source": sample["source"],
                "true": y_true[i],
                "pred": emb_pred[i],
                "score": round(emb_scores[i], 4),
                "text": sample["text"][:200] + "...",
                "notes": sample.get("notes", "")
            })
        
        # TF-IDF errors
        if tfidf_pred[i] == "high" and y_true[i] == "low":
            errors["tfidf"]["false_positives"].append({
                "id": sample["id"],
                "source": sample["source"],
                "true": y_true[i],
                "pred": tfidf_pred[i],
                "score": round(tfidf_scores[i], 4),
                "text": sample["text"][:200] + "..."
            })
        elif tfidf_pred[i] == "low" and y_true[i] == "high":
            errors["tfidf"]["false_negatives"].append({
                "id": sample["id"],
                "source": sample["source"],
                "true": y_true[i],
                "pred": tfidf_pred[i],
                "score": round(tfidf_scores[i], 4),
                "text": sample["text"][:200] + "..."
            })
    
    return errors


def run_comparison(data_path: str = "evaluation/labeled_risks.json"):
    """Run full comparison analysis."""
    print("\n" + "="*60)
    print("TF-IDF vs EMBEDDINGS COMPARISON")
    print("="*60)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    
    print(f"\nScoring {len(samples)} samples...")
    
    # Get scores
    print("  Computing TF-IDF scores...")
    tfidf_scores = score_with_tfidf(texts)
    
    print("  Computing embedding scores...")
    emb_scores = score_with_embeddings(texts)
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(samples, tfidf_scores, emb_scores)
    
    # Identify errors
    print("\nAnalyzing errors...")
    errors = identify_errors(samples, tfidf_scores, emb_scores)
    
    print(f"\nEmbedding Errors:")
    print(f"  False Positives (low→high): {len(errors['embedding']['false_positives'])}")
    print(f"  False Negatives (high→low): {len(errors['embedding']['false_negatives'])}")
    
    print(f"\nTF-IDF Errors:")
    print(f"  False Positives (low→high): {len(errors['tfidf']['false_positives'])}")
    print(f"  False Negatives (high→low): {len(errors['tfidf']['false_negatives'])}")
    
    # Save error analysis
    with open("evaluation/error_analysis.json", "w") as f:
        json.dump(errors, f, indent=2)
    print(f"\n✅ Error analysis saved to evaluation/error_analysis.json")
    
    return {
        "tfidf_scores": tfidf_scores,
        "emb_scores": emb_scores,
        "errors": errors
    }


if __name__ == "__main__":
    run_comparison()

