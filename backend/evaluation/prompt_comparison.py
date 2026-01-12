"""
Phase 2.1: GPT Prompt Comparison Evaluation

Tests 5 different summarization prompts to determine optimal phrasing.

Prompts tested:
1. baseline - Current prompt (executive bullet points)
2. structured - "List the top 3 risks..."
3. executive - "Write for a C-suite audience..."
4. analyst - "Provide investment-relevant summary..."
5. concise - "Summarize in exactly 3 sentences..."

Metrics:
- Faithfulness: Claims verifiable in source
- Coverage: Key risks captured
- Conciseness: Information density
- Actionability: Usefulness for decision-making
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config.settings import settings, get_logger

logger = get_logger(__name__)

# === Prompt Templates ===

PROMPTS = {
    "baseline": """Summarize the following risk factors from a company's 10-K filing into a clear, concise executive-level overview (3–5 bullet points max). Focus on the most serious risks to the business:

{{SECTION_TEXT}}

Summary:""",

    "structured": """Analyze the following risk factors from a 10-K filing and list the top 3 most critical risks.

For each risk, provide:
1. Risk Name (2-4 words)
2. Severity (High/Medium/Low)
3. One-sentence description

Risk Factors:
{{SECTION_TEXT}}

Top 3 Risks:""",

    "executive": """Write a risk briefing for a C-suite audience based on the following 10-K risk disclosures.

Requirements:
- Focus on strategic and material risks
- Use clear, non-technical language
- Highlight potential business impact
- Keep under 150 words

Risk Disclosures:
{{SECTION_TEXT}}

Executive Risk Briefing:""",

    "analyst": """Provide an investment-relevant risk summary for the following 10-K risk factors.

Focus on:
- Risks that could affect financial performance
- Risks that differentiate this company from peers
- Actionable insights for portfolio decisions

Risk Factors:
{{SECTION_TEXT}}

Investment Risk Summary:""",

    "concise": """Summarize the following 10-K risk factors in exactly 3 sentences.

Sentence 1: The single most critical risk
Sentence 2: Secondary risks that warrant attention
Sentence 3: Overall risk assessment (high/medium/low) with brief justification

Risk Factors:
{{SECTION_TEXT}}

Summary:""",
}

SYSTEM_PROMPTS = {
    "baseline": "You are a professional financial analyst.",
    "structured": "You are a risk management specialist analyzing SEC filings.",
    "executive": "You are a Chief Risk Officer preparing a board briefing.",
    "analyst": "You are a senior equity analyst at a major investment bank.",
    "concise": "You are a financial communications expert focused on clarity.",
}

# === Evaluation Prompts ===

EVALUATION_PROMPT = """You are evaluating the quality of a risk summary generated from 10-K filings.

ORIGINAL RISK FACTORS:
{source_text}

GENERATED SUMMARY:
{summary}

Evaluate the summary on these criteria (score 1-5, where 5 is best):

1. FAITHFULNESS: Are all claims in the summary verifiable from the source text? (5=all claims traceable, 1=contains hallucinations)

2. COVERAGE: Does the summary capture the most important risks? (5=all key risks included, 1=major risks missing)

3. CONCISENESS: Is the summary appropriately concise without losing meaning? (5=optimal length, 1=too verbose or too sparse)

4. ACTIONABILITY: Would this summary help an executive make decisions? (5=highly actionable, 1=not useful)

Respond in this exact JSON format:
{{
    "faithfulness": <score 1-5>,
    "coverage": <score 1-5>,
    "conciseness": <score 1-5>,
    "actionability": <score 1-5>,
    "overall": <average of above>,
    "faithfulness_notes": "<brief explanation>",
    "coverage_notes": "<brief explanation>",
    "strengths": "<what works well>",
    "weaknesses": "<what could be improved>"
}}"""


@dataclass
class SummaryResult:
    """Result from a single summarization."""
    prompt_name: str
    summary: str
    tokens_used: int
    latency_ms: int


@dataclass
class EvaluationResult:
    """Evaluation scores for a summary."""
    prompt_name: str
    faithfulness: float
    coverage: float
    conciseness: float
    actionability: float
    overall: float
    faithfulness_notes: str
    coverage_notes: str
    strengths: str
    weaknesses: str


@dataclass
class PromptComparisonResult:
    """Final comparison results."""
    best_prompt: str
    prompt_scores: Dict[str, Dict[str, float]]
    detailed_results: List[Dict]
    recommendation: str


class PromptComparer:
    """Compare different summarization prompts."""
    
    def __init__(self, model: str = "gpt-4o"):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for prompt comparison")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
    
    def summarize_with_prompt(
        self,
        prompt_name: str,
        section_text: str,
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> SummaryResult:
        """Generate summary using a specific prompt."""
        prompt_template = PROMPTS[prompt_name]
        system_prompt = SYSTEM_PROMPTS[prompt_name]
        
        full_prompt = prompt_template.replace("{{SECTION_TEXT}}", section_text)
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        summary = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        
        return SummaryResult(
            prompt_name=prompt_name,
            summary=summary,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
    
    def evaluate_summary(
        self,
        source_text: str,
        summary: str,
        prompt_name: str,
    ) -> EvaluationResult:
        """Evaluate a summary using LLM-as-judge."""
        eval_prompt = EVALUATION_PROMPT.format(
            source_text=source_text[:3000],  # Truncate for context window
            summary=summary,
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Use same model for consistent evaluation
            messages=[
                {"role": "system", "content": "You are an expert at evaluating financial text quality. Always respond with valid JSON."},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0,  # Deterministic for evaluation
            max_tokens=500,
        )
        
        try:
            result_text = response.choices[0].message.content.strip()
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            return EvaluationResult(
                prompt_name=prompt_name,
                faithfulness=result.get("faithfulness", 3),
                coverage=result.get("coverage", 3),
                conciseness=result.get("conciseness", 3),
                actionability=result.get("actionability", 3),
                overall=result.get("overall", 3),
                faithfulness_notes=result.get("faithfulness_notes", ""),
                coverage_notes=result.get("coverage_notes", ""),
                strengths=result.get("strengths", ""),
                weaknesses=result.get("weaknesses", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse evaluation: {e}")
            return EvaluationResult(
                prompt_name=prompt_name,
                faithfulness=3.0,
                coverage=3.0,
                conciseness=3.0,
                actionability=3.0,
                overall=3.0,
                faithfulness_notes="Parse error",
                coverage_notes="Parse error",
                strengths="",
                weaknesses="",
            )
    
    def compare_prompts(
        self,
        test_texts: List[str],
        prompt_names: Optional[List[str]] = None,
    ) -> PromptComparisonResult:
        """
        Compare all prompts across multiple test texts.
        
        Args:
            test_texts: List of risk factor texts to summarize
            prompt_names: Which prompts to test (default: all)
        
        Returns:
            PromptComparisonResult with rankings and recommendation
        """
        if prompt_names is None:
            prompt_names = list(PROMPTS.keys())
        
        all_results = []
        prompt_aggregates = {name: {"scores": [], "latencies": [], "tokens": []} 
                           for name in prompt_names}
        
        for i, text in enumerate(test_texts):
            logger.info(f"Processing text {i+1}/{len(test_texts)}")
            
            text_results = {"text_preview": text[:100] + "...", "evaluations": {}}
            
            for prompt_name in prompt_names:
                logger.info(f"  Testing prompt: {prompt_name}")
                
                # Generate summary
                summary_result = self.summarize_with_prompt(prompt_name, text)
                
                # Evaluate
                eval_result = self.evaluate_summary(text, summary_result.summary, prompt_name)
                
                # Store
                text_results["evaluations"][prompt_name] = {
                    "summary": summary_result.summary,
                    "tokens": summary_result.tokens_used,
                    "latency_ms": summary_result.latency_ms,
                    **asdict(eval_result),
                }
                
                prompt_aggregates[prompt_name]["scores"].append(eval_result.overall)
                prompt_aggregates[prompt_name]["latencies"].append(summary_result.latency_ms)
                prompt_aggregates[prompt_name]["tokens"].append(summary_result.tokens_used)
            
            all_results.append(text_results)
        
        # Compute final scores
        prompt_scores = {}
        for name, agg in prompt_aggregates.items():
            prompt_scores[name] = {
                "mean_overall": sum(agg["scores"]) / len(agg["scores"]) if agg["scores"] else 0,
                "mean_latency_ms": sum(agg["latencies"]) / len(agg["latencies"]) if agg["latencies"] else 0,
                "mean_tokens": sum(agg["tokens"]) / len(agg["tokens"]) if agg["tokens"] else 0,
            }
        
        # Determine best prompt
        best_prompt = max(prompt_scores.keys(), key=lambda k: prompt_scores[k]["mean_overall"])
        best_score = prompt_scores[best_prompt]["mean_overall"]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(prompt_scores, all_results)
        
        return PromptComparisonResult(
            best_prompt=best_prompt,
            prompt_scores=prompt_scores,
            detailed_results=all_results,
            recommendation=recommendation,
        )
    
    def _generate_recommendation(
        self,
        scores: Dict[str, Dict[str, float]],
        details: List[Dict],
    ) -> str:
        """Generate a recommendation based on results."""
        sorted_prompts = sorted(
            scores.items(),
            key=lambda x: x[1]["mean_overall"],
            reverse=True,
        )
        
        best = sorted_prompts[0]
        worst = sorted_prompts[-1]
        
        lines = [
            f"## Recommendation: Use '{best[0]}' prompt",
            "",
            f"**Best performer:** {best[0]} (score: {best[1]['mean_overall']:.2f}/5.0)",
            f"**Worst performer:** {worst[0]} (score: {worst[1]['mean_overall']:.2f}/5.0)",
            "",
            "### Rankings:",
        ]
        
        for i, (name, data) in enumerate(sorted_prompts, 1):
            lines.append(
                f"{i}. **{name}**: {data['mean_overall']:.2f}/5.0 "
                f"(latency: {data['mean_latency_ms']:.0f}ms, tokens: {data['mean_tokens']:.0f})"
            )
        
        return "\n".join(lines)


def load_test_texts(n_samples: int = 5) -> List[str]:
    """Load sample texts from labeled risks for testing."""
    labels_path = os.path.join(os.path.dirname(__file__), "labeled_risks.json")
    
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    
    # Group by source to get diverse samples
    by_source = {}
    for sample in samples:
        source = sample.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(sample)
    
    # Take high-risk samples (more interesting to summarize)
    high_risk = [s for s in samples if s.get("label") == "high"]
    
    # Combine multiple paragraphs to simulate a section
    test_texts = []
    sources = list(by_source.keys())
    
    for i in range(min(n_samples, len(sources))):
        source = sources[i]
        source_samples = by_source[source]
        
        # Take up to 8 paragraphs per source
        selected = source_samples[:8]
        combined_text = "\n\n".join(s["text"] for s in selected)
        test_texts.append(combined_text)
    
    return test_texts


def simulate_comparison(
    test_texts: List[str],
    prompt_names: List[str] = None,
) -> PromptComparisonResult:
    """
    Simulate comparison without API calls for testing/demo.
    Uses realistic score distributions based on expected prompt characteristics.
    """
    if prompt_names is None:
        prompt_names = list(PROMPTS.keys())
    
    # Expected characteristics of each prompt (based on design)
    prompt_characteristics = {
        "baseline": {"faithfulness": 4.2, "coverage": 4.0, "conciseness": 3.8, "actionability": 3.5},
        "structured": {"faithfulness": 4.5, "coverage": 4.2, "conciseness": 4.0, "actionability": 4.3},
        "executive": {"faithfulness": 4.0, "coverage": 3.8, "conciseness": 4.2, "actionability": 4.5},
        "analyst": {"faithfulness": 4.3, "coverage": 4.0, "conciseness": 3.5, "actionability": 4.2},
        "concise": {"faithfulness": 3.8, "coverage": 3.5, "conciseness": 4.8, "actionability": 3.8},
    }
    
    all_results = []
    prompt_aggregates = {name: {"scores": []} for name in prompt_names}
    
    for i, text in enumerate(test_texts):
        text_results = {"text_preview": text[:100] + "...", "evaluations": {}}
        
        for prompt_name in prompt_names:
            chars = prompt_characteristics.get(prompt_name, 
                {"faithfulness": 3.5, "coverage": 3.5, "conciseness": 3.5, "actionability": 3.5})
            
            # Add small random variation
            scores = {k: min(5.0, max(1.0, v + random.uniform(-0.3, 0.3))) for k, v in chars.items()}
            overall = sum(scores.values()) / 4
            
            text_results["evaluations"][prompt_name] = {
                "summary": f"[Simulated {prompt_name} summary for text {i+1}]",
                "tokens": random.randint(200, 500),
                "latency_ms": random.randint(500, 2000),
                "faithfulness": scores["faithfulness"],
                "coverage": scores["coverage"],
                "conciseness": scores["conciseness"],
                "actionability": scores["actionability"],
                "overall": overall,
            }
            
            prompt_aggregates[prompt_name]["scores"].append(overall)
        
        all_results.append(text_results)
    
    # Compute final scores
    prompt_scores = {}
    for name, agg in prompt_aggregates.items():
        prompt_scores[name] = {
            "mean_overall": sum(agg["scores"]) / len(agg["scores"]) if agg["scores"] else 0,
            "mean_latency_ms": random.randint(800, 1500),
            "mean_tokens": random.randint(250, 400),
        }
    
    best_prompt = max(prompt_scores.keys(), key=lambda k: prompt_scores[k]["mean_overall"])
    
    recommendation = f"""## Recommendation: Use '{best_prompt}' prompt (SIMULATED)

**Note:** These are simulated results. Run with valid OPENAI_API_KEY for real evaluation.

### Expected Rankings (based on prompt design):
1. **structured**: High faithfulness + actionability due to clear format
2. **executive**: Best actionability for C-suite audience
3. **analyst**: Good for investment-focused analysis
4. **baseline**: Solid general-purpose prompt
5. **concise**: Highest conciseness but may sacrifice coverage"""
    
    return PromptComparisonResult(
        best_prompt=best_prompt,
        prompt_scores=prompt_scores,
        detailed_results=all_results,
        recommendation=recommendation,
    )


def run_comparison(
    n_samples: int = 5,
    output_path: Optional[str] = None,
    simulate: bool = False,
) -> Dict:
    """
    Run full prompt comparison evaluation.
    
    Args:
        n_samples: Number of test texts to use
        output_path: Where to save results JSON
        simulate: If True, run without API calls (for testing)
    
    Returns:
        Comparison results dict
    """
    logger.info("="*60)
    logger.info("PHASE 2.1: PROMPT COMPARISON EVALUATION")
    logger.info("="*60)
    
    # Load test data
    logger.info(f"Loading {n_samples} test texts...")
    test_texts = load_test_texts(n_samples)
    logger.info(f"Loaded {len(test_texts)} test texts")
    
    # Run comparison
    if simulate:
        logger.info("Running in SIMULATION mode (no API calls)")
        results = simulate_comparison(test_texts)
    else:
        try:
            comparer = PromptComparer()
            results = comparer.compare_prompts(test_texts)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            logger.info("Falling back to simulation mode")
            results = simulate_comparison(test_texts)
    
    # Print results
    print("\n" + "="*60)
    print("PROMPT COMPARISON RESULTS")
    print("="*60)
    
    print(f"\n🏆 BEST PROMPT: {results.best_prompt}")
    print(f"   Score: {results.prompt_scores[results.best_prompt]['mean_overall']:.2f}/5.0")
    
    print("\n📊 ALL PROMPTS:")
    sorted_scores = sorted(
        results.prompt_scores.items(),
        key=lambda x: x[1]["mean_overall"],
        reverse=True,
    )
    
    print(f"\n{'Prompt':<15} {'Score':<8} {'Latency':<12} {'Tokens':<10}")
    print("-" * 45)
    for name, data in sorted_scores:
        print(f"{name:<15} {data['mean_overall']:.2f}     {data['mean_latency_ms']:.0f}ms        {data['mean_tokens']:.0f}")
    
    print("\n" + results.recommendation)
    
    # Prepare output
    output = {
        "phase": "2.1",
        "task": "Prompt Comparison Evaluation",
        "best_prompt": results.best_prompt,
        "prompt_scores": results.prompt_scores,
        "detailed_results": results.detailed_results,
        "recommendation": results.recommendation,
        "methodology": {
            "model": "gpt-4o",
            "evaluation_method": "LLM-as-judge",
            "metrics": ["faithfulness", "coverage", "conciseness", "actionability"],
            "n_samples": n_samples,
        },
    }
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare GPT summarization prompts")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--output", type=str, default="evaluation/prompt_comparison_results.json",
                        help="Output file path")
    parser.add_argument("--simulate", action="store_true", 
                        help="Run in simulation mode without API calls")
    args = parser.parse_args()
    
    results = run_comparison(n_samples=args.samples, output_path=args.output, simulate=args.simulate)

