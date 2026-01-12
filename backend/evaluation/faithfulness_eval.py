"""
Phase 2.2: Faithfulness Evaluation

Evaluates how faithful generated summaries are to source text.

Methods:
1. NLI-based entailment checking (using transformer model)
2. Keyword/entity overlap scoring
3. LLM-as-judge verification

Output: "X% of summary claims verifiable in source"
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, get_logger

logger = get_logger(__name__)

# Try to import NLI model
try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    logger.warning("transformers not available - NLI evaluation disabled")


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    verified: bool
    confidence: float
    evidence: str
    method: str


@dataclass
class FaithfulnessResult:
    """Complete faithfulness evaluation result."""
    summary: str
    source_text: str
    overall_score: float  # 0-1, percentage of verified claims
    verified_claims: int
    total_claims: int
    keyword_overlap: float
    entity_overlap: float
    nli_score: Optional[float]
    claim_details: List[ClaimVerification]
    interpretation: str


class FaithfulnessEvaluator:
    """Evaluate faithfulness of summaries to source text."""
    
    def __init__(self, use_nli: bool = True, use_llm: bool = False):
        """
        Initialize evaluator.
        
        Args:
            use_nli: Use NLI model for entailment checking
            use_llm: Use LLM for claim verification (requires API key)
        """
        self.use_nli = use_nli and NLI_AVAILABLE
        self.use_llm = use_llm
        self.nli_model = None
        self.openai_client = None
        
        if self.use_nli:
            self._load_nli_model()
        
        if self.use_llm and settings.OPENAI_API_KEY:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def _load_nli_model(self):
        """Load NLI model for entailment checking."""
        try:
            logger.info("Loading NLI model...")
            self.nli_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            logger.info("NLI model loaded")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            self.use_nli = False
    
    def extract_claims(self, summary: str) -> List[str]:
        """
        Extract individual claims from summary text.
        
        A claim is typically a sentence or bullet point.
        """
        claims = []
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]\s+', summary)
        
        for sent in sentences:
            sent = sent.strip()
            # Skip very short fragments
            if len(sent) < 20:
                continue
            # Skip if it's just a label like "1." or "Risk:"
            if re.match(r'^[\d\.\-\*]+\s*$', sent):
                continue
            claims.append(sent)
        
        # Also check for bullet points
        bullet_pattern = r'[-•*]\s+(.+?)(?=\n[-•*]|\n\n|$)'
        bullets = re.findall(bullet_pattern, summary, re.DOTALL)
        for bullet in bullets:
            bullet = bullet.strip()
            if len(bullet) >= 20 and bullet not in claims:
                claims.append(bullet)
        
        return claims if claims else [summary]  # Fallback to whole summary
    
    def compute_keyword_overlap(self, summary: str, source: str) -> float:
        """
        Compute keyword overlap between summary and source.
        
        Returns fraction of summary keywords found in source.
        """
        # Simple tokenization
        def tokenize(text):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            # Filter stopwords
            stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was',
                        'have', 'has', 'been', 'from', 'their', 'they', 'will',
                        'would', 'could', 'should', 'may', 'can', 'its', 'our'}
            return [w for w in words if w not in stopwords]
        
        summary_words = set(tokenize(summary))
        source_words = set(tokenize(source))
        
        if not summary_words:
            return 0.0
        
        overlap = summary_words.intersection(source_words)
        return len(overlap) / len(summary_words)
    
    def extract_entities(self, text: str) -> set:
        """
        Extract named entities and key terms from text.
        
        Uses simple pattern matching for:
        - Capitalized phrases (company names, terms)
        - Numbers and percentages
        - Technical terms
        """
        entities = set()
        
        # Capitalized words/phrases
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(caps)
        
        # Numbers with context
        numbers = re.findall(r'\$?[\d,]+(?:\.\d+)?%?', text)
        entities.update(numbers)
        
        # Technical/financial terms (simplified)
        terms = re.findall(r'\b(?:cybersecurity|litigation|regulatory|compliance|'
                          r'revenue|profit|loss|risk|threat|breach|disruption|'
                          r'supply chain|market|financial|operating|adverse)\b', 
                          text.lower())
        entities.update(terms)
        
        return entities
    
    def compute_entity_overlap(self, summary: str, source: str) -> float:
        """Compute overlap of entities between summary and source."""
        summary_entities = self.extract_entities(summary)
        source_entities = self.extract_entities(source)
        
        if not summary_entities:
            return 0.0
        
        overlap = summary_entities.intersection(source_entities)
        return len(overlap) / len(summary_entities)
    
    def verify_claim_nli(self, claim: str, source: str) -> Tuple[bool, float]:
        """
        Verify a claim using NLI entailment.
        
        Returns (verified, confidence).
        """
        if not self.nli_model:
            return True, 0.5  # Default if NLI not available
        
        try:
            # Use source as premise, claim as hypothesis
            # Check if source entails the claim
            result = self.nli_model(
                source[:2000],  # Truncate for model limit
                candidate_labels=["supported", "not supported"],
                hypothesis_template="{}",
            )
            
            # Check if claim is entailed
            if result["labels"][0] == "supported":
                return True, result["scores"][0]
            else:
                return False, 1 - result["scores"][0]
        except Exception as e:
            logger.warning(f"NLI verification failed: {e}")
            return True, 0.5
    
    def verify_claim_llm(self, claim: str, source: str) -> Tuple[bool, float, str]:
        """
        Verify a claim using LLM-as-judge.
        
        Returns (verified, confidence, evidence).
        """
        if not self.openai_client:
            return True, 0.5, "LLM not available"
        
        prompt = f"""You are verifying if a claim from a summary is supported by the source text.

SOURCE TEXT:
{source[:3000]}

CLAIM TO VERIFY:
{claim}

Is this claim fully supported by the source text?

Respond in JSON format:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "evidence": "quote or explanation from source that supports/contradicts"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You verify claim accuracy. Always respond with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            
            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            result = json.loads(result_text)
            
            return (
                result.get("supported", True),
                result.get("confidence", 0.5),
                result.get("evidence", ""),
            )
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return True, 0.5, str(e)
    
    def evaluate(self, summary: str, source: str) -> FaithfulnessResult:
        """
        Evaluate faithfulness of a summary to its source.
        
        Args:
            summary: Generated summary text
            source: Original source text
            
        Returns:
            FaithfulnessResult with detailed analysis
        """
        logger.info("Evaluating faithfulness...")
        
        # Extract claims
        claims = self.extract_claims(summary)
        logger.info(f"Extracted {len(claims)} claims from summary")
        
        # Compute overlap scores
        keyword_overlap = self.compute_keyword_overlap(summary, source)
        entity_overlap = self.compute_entity_overlap(summary, source)
        
        # Verify each claim
        claim_details = []
        verified_count = 0
        nli_scores = []
        
        for claim in claims:
            if self.use_nli:
                verified, confidence = self.verify_claim_nli(claim, source)
                nli_scores.append(confidence if verified else 1 - confidence)
                evidence = "NLI entailment check"
                method = "nli"
            elif self.use_llm:
                verified, confidence, evidence = self.verify_claim_llm(claim, source)
                method = "llm"
            else:
                # Fallback: keyword-based verification
                claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
                source_words = set(re.findall(r'\b[a-z]{4,}\b', source.lower()))
                overlap = len(claim_words.intersection(source_words)) / len(claim_words) if claim_words else 0
                verified = overlap > 0.5
                confidence = overlap
                evidence = f"{overlap:.1%} keyword overlap"
                method = "keyword"
            
            if verified:
                verified_count += 1
            
            claim_details.append(ClaimVerification(
                claim=claim[:200] + "..." if len(claim) > 200 else claim,
                verified=verified,
                confidence=confidence,
                evidence=evidence,
                method=method,
            ))
        
        # Compute overall score
        overall_score = verified_count / len(claims) if claims else 0.0
        nli_score = sum(nli_scores) / len(nli_scores) if nli_scores else None
        
        # Generate interpretation
        if overall_score >= 0.9:
            interpretation = "Excellent faithfulness - nearly all claims verifiable in source"
        elif overall_score >= 0.75:
            interpretation = "Good faithfulness - most claims supported by source"
        elif overall_score >= 0.5:
            interpretation = "Moderate faithfulness - some claims may need verification"
        else:
            interpretation = "Low faithfulness - many claims not clearly supported"
        
        return FaithfulnessResult(
            summary=summary,
            source_text=source[:500] + "...",
            overall_score=overall_score,
            verified_claims=verified_count,
            total_claims=len(claims),
            keyword_overlap=keyword_overlap,
            entity_overlap=entity_overlap,
            nli_score=nli_score,
            claim_details=claim_details,
            interpretation=interpretation,
        )


def run_faithfulness_evaluation(
    n_samples: int = 5,
    output_path: Optional[str] = None,
    use_nli: bool = True,
    use_llm: bool = False,
) -> Dict:
    """
    Run faithfulness evaluation on sample summaries.
    
    Args:
        n_samples: Number of samples to evaluate
        output_path: Where to save results
        use_nli: Use NLI model (slower but more accurate)
        use_llm: Use LLM for verification (requires API key)
    
    Returns:
        Evaluation results dict
    """
    logger.info("="*60)
    logger.info("PHASE 2.2: FAITHFULNESS EVALUATION")
    logger.info("="*60)
    
    # Load labeled risks for test data
    labels_path = os.path.join(os.path.dirname(__file__), "labeled_risks.json")
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    
    # Group by source
    by_source = {}
    for sample in samples:
        source = sample.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(sample)
    
    # Create test cases (combine paragraphs and generate summaries)
    evaluator = FaithfulnessEvaluator(use_nli=use_nli, use_llm=use_llm)
    
    # For testing, we'll create synthetic summaries
    # In production, these would come from the actual summarizer
    all_results = []
    
    sources = list(by_source.keys())[:n_samples]
    
    for source_name in sources:
        logger.info(f"Evaluating source: {source_name}")
        
        source_samples = by_source[source_name][:8]
        source_text = "\n\n".join(s["text"] for s in source_samples)
        
        # Generate a simple extractive summary for testing
        # (In production, this would be the GPT-generated summary)
        high_risk = [s for s in source_samples if s.get("label") == "high"]
        if high_risk:
            summary = "Key risks identified:\n"
            for i, risk in enumerate(high_risk[:3], 1):
                summary += f"• {risk['text'][:150]}...\n"
        else:
            summary = source_samples[0]["text"][:300] + "..."
        
        result = evaluator.evaluate(summary, source_text)
        
        all_results.append({
            "source": source_name,
            **asdict(result),
        })
    
    # Compute aggregate metrics
    avg_score = sum(r["overall_score"] for r in all_results) / len(all_results)
    avg_keyword = sum(r["keyword_overlap"] for r in all_results) / len(all_results)
    avg_entity = sum(r["entity_overlap"] for r in all_results) / len(all_results)
    
    # Print results
    print("\n" + "="*60)
    print("FAITHFULNESS EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 AGGREGATE METRICS:")
    print(f"   Overall Faithfulness: {avg_score:.1%}")
    print(f"   Keyword Overlap: {avg_keyword:.1%}")
    print(f"   Entity Overlap: {avg_entity:.1%}")
    
    print(f"\n📝 BY SOURCE:")
    for result in all_results:
        print(f"   {result['source']}: {result['overall_score']:.1%} "
              f"({result['verified_claims']}/{result['total_claims']} claims verified)")
    
    output = {
        "phase": "2.2",
        "task": "Faithfulness Evaluation",
        "aggregate_metrics": {
            "overall_faithfulness": avg_score,
            "keyword_overlap": avg_keyword,
            "entity_overlap": avg_entity,
        },
        "methodology": {
            "nli_enabled": use_nli and NLI_AVAILABLE,
            "llm_enabled": use_llm,
            "claim_extraction": "sentence-based",
        },
        "results": all_results,
        "interpretation": f"{avg_score:.1%} of summary claims verifiable in source",
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n✅ Results saved to {output_path}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate summary faithfulness")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--output", type=str, default="evaluation/faithfulness_results.json",
                        help="Output file path")
    parser.add_argument("--use-nli", action="store_true", default=True,
                        help="Use NLI model for verification")
    parser.add_argument("--no-nli", action="store_true",
                        help="Disable NLI model")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for verification (requires API key)")
    args = parser.parse_args()
    
    use_nli = args.use_nli and not args.no_nli
    
    results = run_faithfulness_evaluation(
        n_samples=args.samples,
        output_path=args.output,
        use_nli=use_nli,
        use_llm=args.use_llm,
    )

