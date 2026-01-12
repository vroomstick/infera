"""
Embedding Space Visualization and Cluster Analysis.

Phase 1.3: Generate t-SNE/UMAP plots of paragraph embeddings
Phase 1.4: K-means clustering to identify natural risk categories

Visualizations:
- Color by risk score (continuous)
- Color by company (categorical)  
- Color by label (high/medium/low)
- Color by cluster (k-means)
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter

# Try UMAP, fall back to t-SNE only
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed, using t-SNE only")


# Risk prompt for scoring
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Company colors
COMPANY_COLORS = {
    'AAPL': '#555555',   # Gray
    'TSLA': '#E31937',   # Red
    'MSFT': '#00A4EF',   # Blue
    'NVDA': '#76B900',   # Green
    'AMZN': '#FF9900',   # Orange
    'GOOGL': '#4285F4',  # Google Blue
}

# Label colors
LABEL_COLORS = {
    'high': '#D32F2F',    # Red
    'medium': '#FFA000',  # Amber
    'low': '#388E3C',     # Green
}


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def compute_embeddings_and_scores(
    texts: List[str],
    model_name: str = "ProsusAI/finbert"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute embeddings and risk scores for all texts.
    
    Returns:
        embeddings: (n, dim) array
        scores: (n,) array
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Computing embeddings for {len(texts)} texts...")
    all_texts = [RISK_PROMPT] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    
    risk_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    # Compute scores
    similarities = cosine_similarity(text_embeddings, risk_embedding).flatten()
    scores = np.clip(similarities, 0, 1)
    
    return text_embeddings, scores


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: (n, dim) array
        method: 'tsne' or 'umap'
        
    Returns:
        (n, 2) array for plotting
    """
    print(f"Reducing dimensions with {method.upper()}...")
    
    if method == "umap" and HAS_UMAP:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=random_state
        )
    else:
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,
            random_state=random_state,
            max_iter=1000
        )
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Run k-means clustering on embeddings.
    
    Returns:
        cluster_labels: (n,) array of cluster assignments
        kmeans: fitted KMeans model
    """
    print(f"Running k-means with {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, kmeans


def plot_by_score(
    coords: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    title: str = "Embedding Space by Risk Score"
):
    """Plot embeddings colored by risk score."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=scores,
        cmap='RdYlGn_r',  # Red = high risk, Green = low risk
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Score', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_company(
    coords: np.ndarray,
    companies: List[str],
    output_path: str,
    title: str = "Embedding Space by Company"
):
    """Plot embeddings colored by company."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for company in COMPANY_COLORS:
        mask = np.array([c == company for c in companies])
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=COMPANY_COLORS[company],
                label=f"{company} ({mask.sum()})",
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_label(
    coords: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "Embedding Space by Risk Label"
):
    """Plot embeddings colored by risk label."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in ['low', 'medium', 'high']:
        mask = np.array([l == label for l in labels])
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=LABEL_COLORS[label],
                label=f"{label.capitalize()} ({mask.sum()})",
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_cluster(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    title: str = "Embedding Space by Cluster"
):
    """Plot embeddings colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n_clusters = len(set(cluster_labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]],
            label=f"Cluster {i} ({mask.sum()})",
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_clusters(
    samples: List[Dict],
    cluster_labels: np.ndarray,
    n_clusters: int
) -> Dict:
    """
    Analyze cluster composition and identify themes.
    
    Returns cluster analysis with:
    - Label distribution per cluster
    - Company distribution per cluster
    - Sample texts from each cluster
    """
    analysis = {}
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_samples = [s for s, m in zip(samples, mask) if m]
        
        # Label distribution
        label_counts = Counter(s['label'] for s in cluster_samples)
        
        # Company distribution
        company_counts = Counter(s['source'] for s in cluster_samples)
        
        # Dominant characteristics
        dominant_label = label_counts.most_common(1)[0][0] if label_counts else 'unknown'
        dominant_company = company_counts.most_common(1)[0][0] if company_counts else 'unknown'
        
        # Sample texts (first 3)
        sample_texts = [s['text'][:100] + "..." for s in cluster_samples[:3]]
        
        # Common words (simple analysis)
        all_text = " ".join(s['text'].lower() for s in cluster_samples)
        words = [w for w in all_text.split() if len(w) > 5 and w.isalpha()]
        common_words = [w for w, c in Counter(words).most_common(10)]
        
        analysis[f"cluster_{i}"] = {
            "size": int(mask.sum()),
            "label_distribution": dict(label_counts),
            "company_distribution": dict(company_counts),
            "dominant_label": dominant_label,
            "dominant_company": dominant_company,
            "common_words": common_words,
            "sample_texts": sample_texts,
        }
    
    return analysis


def run_visualization_pipeline(
    data_path: str = "evaluation/labeled_risks.json",
    output_dir: str = "evaluation/plots",
    model_name: str = "ProsusAI/finbert",
    n_clusters: int = 8,
    method: str = "tsne"
) -> Dict:
    """
    Run the full visualization and clustering pipeline.
    """
    print("\n" + "="*70)
    print("EMBEDDING VISUALIZATION & CLUSTER ANALYSIS")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    companies = [s["source"] for s in samples]
    
    print(f"\nLoaded {len(samples)} samples")
    
    # Compute embeddings and scores
    embeddings, scores = compute_embeddings_and_scores(texts, model_name)
    
    # Reduce dimensions
    coords = reduce_dimensions(embeddings, method=method)
    
    # Run clustering
    cluster_labels, kmeans = run_kmeans_clustering(embeddings, n_clusters=n_clusters)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_by_score(
        coords, scores,
        os.path.join(output_dir, "embedding_by_score.png"),
        title=f"Paragraph Embeddings by Risk Score ({method.upper()})"
    )
    
    plot_by_company(
        coords, companies,
        os.path.join(output_dir, "embedding_by_company.png"),
        title=f"Paragraph Embeddings by Company ({method.upper()})"
    )
    
    plot_by_label(
        coords, labels,
        os.path.join(output_dir, "embedding_by_label.png"),
        title=f"Paragraph Embeddings by Risk Label ({method.upper()})"
    )
    
    plot_by_cluster(
        coords, cluster_labels,
        os.path.join(output_dir, "embedding_by_cluster.png"),
        title=f"Paragraph Embeddings by Cluster (k={n_clusters})"
    )
    
    # Analyze clusters
    print("\nAnalyzing clusters...")
    cluster_analysis = analyze_clusters(samples, cluster_labels, n_clusters)
    
    # Print cluster summary
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*70)
    
    for cluster_name, info in cluster_analysis.items():
        print(f"\n{cluster_name.upper()} (n={info['size']})")
        print(f"  Labels: {info['label_distribution']}")
        print(f"  Dominant: {info['dominant_label']} risk")
        print(f"  Top words: {', '.join(info['common_words'][:5])}")
    
    # Save analysis
    results = {
        "n_samples": len(samples),
        "n_clusters": n_clusters,
        "method": method,
        "model": model_name,
        "cluster_analysis": cluster_analysis,
        "plots_generated": [
            "embedding_by_score.png",
            "embedding_by_company.png", 
            "embedding_by_label.png",
            "embedding_by_cluster.png",
        ]
    }
    
    output_file = os.path.join(output_dir, "cluster_analysis.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Analysis saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding visualization and clustering")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output-dir", default="evaluation/plots", help="Output directory for plots")
    parser.add_argument("--model", default="ProsusAI/finbert", help="Embedding model")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne", help="Dimensionality reduction method")
    args = parser.parse_args()
    
    results = run_visualization_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model,
        n_clusters=args.n_clusters,
        method=args.method,
    )

