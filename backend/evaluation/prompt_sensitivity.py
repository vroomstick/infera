"""
Prompt Sensitivity Analysis (Phase 3.1)

Tests 10 different risk prompts to measure ranking stability.
Computes pairwise Spearman correlation between prompt rankings.

Deliverable: "Rankings are robust across prompts (rho > 0.9)" or "Prompt X is most stable"
"""

import os
import sys
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from itertools import combinations

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# 10 different risk prompts to test
RISK_PROMPTS = {
    "baseline": """
        Significant business risks including lawsuits, litigation, regulatory compliance,
        cybersecurity threats, data breaches, supply chain disruption, economic downturn,
        inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
        competitive pressure, market volatility, and operational failure.
    """,
    "critical_threat": """
        Critical business threat that could cause severe damage to the company's 
        operations, reputation, financial performance, or long-term viability.
    """,
    "material_adverse": """
        Material adverse condition that could significantly harm the company's 
        revenue, profitability, stock price, or ability to conduct business.
    """,
    "operational_concern": """
        Significant operational concern affecting business continuity, 
        supply chain reliability, workforce stability, or process execution.
    """,
    "major_risk_factor": """
        Major risk factor disclosed in SEC filings that investors should consider
        when evaluating the company's future performance and prospects.
    """,
    "financial_risk": """
        High-impact financial risk including credit exposure, liquidity constraints,
        currency fluctuations, interest rate sensitivity, and capital market access.
    """,
    "business_disruption": """
        Severe business disruption from external or internal factors that could
        materially impair the company's ability to operate effectively.
    """,
    "negative_impact": """
        Substantial negative impact on business operations, financial results,
        competitive position, or stakeholder value creation.
    """,
    "corporate_threat": """
        Serious corporate threat to shareholder value, market position,
        regulatory standing, or organizational sustainability.
    """,
    "risk_exposure": """
        Elevated risk exposure requiring management attention and potential
        mitigation strategies to protect business interests.
    """,
}


@dataclass
class PromptResult:
    """Results for a single prompt."""
    name: str
    prompt_text: str
    scores: List[float]
    rankings: List[int]


@dataclass
class PairwiseCorrelation:
    """Correlation between two prompts."""
    prompt_a: str
    prompt_b: str
    spearman_rho: float
    p_value: float


def load_labeled_data(path: str = "evaluation/labeled_risks.json") -> Dict:
    """Load the labeled evaluation dataset."""
    with open(path, "r") as f:
        return json.load(f)


def score_with_prompt(
    texts: List[str],
    prompt: str,
    model: SentenceTransformer,
) -> Tuple[List[float], List[int]]:
    """
    Score texts using a given prompt.
    
    Returns: (scores, rankings)
    """
    # Embed prompt and texts
    all_texts = [prompt] + texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    
    prompt_embedding = embeddings[0:1]
    text_embeddings = embeddings[1:]
    
    # Compute similarities
    similarities = cosine_similarity(text_embeddings, prompt_embedding).flatten()
    scores = np.clip(similarities, 0, 1).tolist()
    
    # Compute rankings (1 = highest score)
    rankings = (np.argsort(np.argsort(-np.array(scores))) + 1).tolist()
    
    return scores, rankings


def run_prompt_sensitivity_analysis(
    data_path: str = "evaluation/labeled_risks.json",
    model_name: str = "ProsusAI/finbert",
) -> Dict:
    """Run prompt sensitivity analysis."""
    print("\n" + "="*70)
    print("PROMPT SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Load data
    data = load_labeled_data(data_path)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    
    print(f"\nDataset: {len(samples)} samples")
    print(f"Testing {len(RISK_PROMPTS)} prompts")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Score with each prompt
    print("\nScoring with each prompt...")
    results = {}
    
    for name, prompt in RISK_PROMPTS.items():
        scores, rankings = score_with_prompt(texts, prompt, model)
        results[name] = PromptResult(
            name=name,
            prompt_text=prompt.strip(),
            scores=scores,
            rankings=rankings,
        )
        
        # Print score stats
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {name:<20}: mean={mean_score:.3f}, std={std_score:.3f}")
    
    # Compute pairwise correlations
    print("\n" + "="*70)
    print("PAIRWISE SPEARMAN CORRELATIONS")
    print("="*70)
    
    prompt_names = list(RISK_PROMPTS.keys())
    correlations = []
    correlation_matrix = np.zeros((len(prompt_names), len(prompt_names)))
    
    for i, name_a in enumerate(prompt_names):
        for j, name_b in enumerate(prompt_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            elif i < j:
                scores_a = results[name_a].scores
                scores_b = results[name_b].scores
                rho, p = spearmanr(scores_a, scores_b)
                
                correlation_matrix[i, j] = rho
                correlation_matrix[j, i] = rho
                
                correlations.append(PairwiseCorrelation(
                    prompt_a=name_a,
                    prompt_b=name_b,
                    spearman_rho=rho,
                    p_value=p,
                ))
    
    # Print correlation stats
    rho_values = [c.spearman_rho for c in correlations]
    mean_rho = np.mean(rho_values)
    min_rho = np.min(rho_values)
    max_rho = np.max(rho_values)
    
    print(f"\nCorrelation statistics:")
    print(f"  Mean ρ: {mean_rho:.3f}")
    print(f"  Min ρ:  {min_rho:.3f}")
    print(f"  Max ρ:  {max_rho:.3f}")
    
    # Find lowest correlations
    sorted_corrs = sorted(correlations, key=lambda x: x.spearman_rho)
    print(f"\nLowest correlations (potential instability):")
    for c in sorted_corrs[:5]:
        print(f"  {c.prompt_a} vs {c.prompt_b}: ρ={c.spearman_rho:.3f}")
    
    # Find most stable prompt (highest average correlation with others)
    prompt_avg_corr = {}
    for name in prompt_names:
        relevant_corrs = [
            c.spearman_rho for c in correlations 
            if c.prompt_a == name or c.prompt_b == name
        ]
        prompt_avg_corr[name] = np.mean(relevant_corrs)
    
    most_stable = max(prompt_avg_corr.items(), key=lambda x: x[1])
    least_stable = min(prompt_avg_corr.items(), key=lambda x: x[1])
    
    print(f"\nPrompt stability ranking (avg ρ with other prompts):")
    for name, avg_rho in sorted(prompt_avg_corr.items(), key=lambda x: -x[1]):
        marker = "⭐" if name == most_stable[0] else "  "
        print(f"  {marker} {name:<20}: {avg_rho:.3f}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if mean_rho >= 0.9:
        conclusion = f"Rankings are highly robust across prompts (mean ρ = {mean_rho:.3f} ≥ 0.9)"
        robust = True
    elif mean_rho >= 0.8:
        conclusion = f"Rankings are reasonably stable across prompts (mean ρ = {mean_rho:.3f})"
        robust = True
    elif mean_rho >= 0.7:
        conclusion = f"Rankings show moderate sensitivity to prompt choice (mean ρ = {mean_rho:.3f})"
        robust = False
    else:
        conclusion = f"Rankings are sensitive to prompt choice (mean ρ = {mean_rho:.3f} < 0.7)"
        robust = False
    
    print(f"\n{conclusion}")
    print(f"Most stable prompt: '{most_stable[0]}' (avg ρ = {most_stable[1]:.3f})")
    print(f"Least stable prompt: '{least_stable[0]}' (avg ρ = {least_stable[1]:.3f})")
    
    # Build output
    output = {
        "summary": {
            "n_prompts": len(RISK_PROMPTS),
            "n_samples": len(samples),
            "mean_correlation": round(mean_rho, 4),
            "min_correlation": round(min_rho, 4),
            "max_correlation": round(max_rho, 4),
            "robust": robust,
            "conclusion": conclusion,
        },
        "prompt_stability": {
            name: round(avg_rho, 4) 
            for name, avg_rho in sorted(prompt_avg_corr.items(), key=lambda x: -x[1])
        },
        "most_stable_prompt": {
            "name": most_stable[0],
            "avg_correlation": round(most_stable[1], 4),
            "text": RISK_PROMPTS[most_stable[0]].strip(),
        },
        "correlations": [
            {
                "prompt_a": c.prompt_a,
                "prompt_b": c.prompt_b,
                "spearman_rho": round(c.spearman_rho, 4),
                "p_value": round(c.p_value, 6),
            }
            for c in sorted(correlations, key=lambda x: -x.spearman_rho)
        ],
        "prompts": {
            name: {
                "text": prompt.strip(),
                "mean_score": round(np.mean(results[name].scores), 4),
                "std_score": round(np.std(results[name].scores), 4),
            }
            for name, prompt in RISK_PROMPTS.items()
        },
    }
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt sensitivity analysis")
    parser.add_argument("--data", default="evaluation/labeled_risks.json", help="Path to labeled data")
    parser.add_argument("--output", default="evaluation/prompt_sensitivity_results.json", help="Output path")
    args = parser.parse_args()
    
    results = run_prompt_sensitivity_analysis(args.data)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

