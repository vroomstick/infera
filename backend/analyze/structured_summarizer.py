"""
Phase 2.3: Structured JSON Output for GPT Summarization

Provides Pydantic schemas for structured summary output, enabling:
- Type-safe summary responses
- Consistent format across summaries
- Easy integration with APIs and agents
- Validation of GPT output

Output format:
{
    "risks": [
        {
            "title": "Cybersecurity Threats",
            "severity": "high",
            "category": "Cybersecurity",
            "description": "...",
            "potential_impact": "...",
            "source_references": ["para_3", "para_7"]
        }
    ],
    "overall_assessment": "...",
    "confidence": 0.85
}
"""

import os
import sys
import json
from typing import List, Optional, Literal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field, field_validator
from config.settings import settings, get_logger

logger = get_logger(__name__)

# === Pydantic Schemas ===

class RiskItem(BaseModel):
    """A single identified risk from the 10-K filing."""
    
    title: str = Field(
        ...,
        description="Short title for the risk (2-5 words)",
        min_length=5,
        max_length=100,
    )
    
    severity: Literal["high", "medium", "low"] = Field(
        ...,
        description="Risk severity: high (material impact), medium (moderate), low (routine)",
    )
    
    category: Literal[
        "Cybersecurity", "Regulatory", "Supply Chain", "Financial",
        "Competitive", "Operational", "Macroeconomic", "Litigation", "Other"
    ] = Field(
        ...,
        description="Risk category classification",
    )
    
    description: str = Field(
        ...,
        description="One-sentence description of the risk",
        min_length=20,
        max_length=500,
    )
    
    potential_impact: str = Field(
        ...,
        description="Potential business impact if risk materializes",
        max_length=300,
    )
    
    source_paragraph: Optional[int] = Field(
        None,
        description="Index of source paragraph (if available)",
    )
    
    @field_validator('title')
    @classmethod
    def title_not_generic(cls, v):
        generic_titles = ["risk", "the risk", "risks", "issue", "problem"]
        if v.lower().strip() in generic_titles:
            raise ValueError("Title must be specific, not generic")
        return v


class StructuredSummary(BaseModel):
    """Complete structured summary of 10-K risk factors."""
    
    ticker: Optional[str] = Field(
        None,
        description="Company ticker symbol",
        pattern=r'^[A-Z]{1,5}$',
    )
    
    risks: List[RiskItem] = Field(
        ...,
        description="List of identified risks, ordered by severity",
        min_length=1,
        max_length=10,
    )
    
    overall_assessment: str = Field(
        ...,
        description="Brief overall risk assessment (2-3 sentences)",
        max_length=500,
    )
    
    risk_level: Literal["high", "moderate", "low"] = Field(
        ...,
        description="Overall risk level for the company",
    )
    
    confidence: float = Field(
        ...,
        description="Confidence in the analysis (0-1)",
        ge=0.0,
        le=1.0,
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of generation",
    )
    
    model: str = Field(
        default="gpt-4o",
        description="Model used for generation",
    )
    
    @field_validator('risks')
    @classmethod
    def ensure_ordered_by_severity(cls, v):
        """Ensure high-severity risks come first."""
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_risks = sorted(v, key=lambda r: severity_order.get(r.severity, 3))
        return sorted_risks
    
    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        lines = [
            f"# Risk Summary{f' - {self.ticker}' if self.ticker else ''}",
            "",
            f"**Overall Risk Level:** {self.risk_level.upper()}",
            f"**Confidence:** {self.confidence:.0%}",
            "",
            "## Key Risks",
            "",
        ]
        
        for i, risk in enumerate(self.risks, 1):
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[risk.severity]
            lines.append(f"### {i}. {risk.title} {severity_emoji}")
            lines.append(f"**Category:** {risk.category} | **Severity:** {risk.severity.upper()}")
            lines.append("")
            lines.append(risk.description)
            lines.append("")
            if risk.potential_impact:
                lines.append(f"*Impact:* {risk.potential_impact}")
                lines.append("")
        
        lines.append("## Overall Assessment")
        lines.append("")
        lines.append(self.overall_assessment)
        lines.append("")
        lines.append(f"---")
        lines.append(f"*Generated: {self.generated_at.isoformat()} | Model: {self.model}*")
        
        return "\n".join(lines)


# === Structured Summarizer ===

STRUCTURED_PROMPT = """Analyze the following risk factors from a 10-K filing and provide a structured JSON summary.

RISK FACTORS:
{{SECTION_TEXT}}

Respond with a valid JSON object matching this schema:
{
    "risks": [
        {
            "title": "Short title (2-5 words)",
            "severity": "high" | "medium" | "low",
            "category": "Cybersecurity" | "Regulatory" | "Supply Chain" | "Financial" | "Competitive" | "Operational" | "Macroeconomic" | "Litigation" | "Other",
            "description": "One-sentence description",
            "potential_impact": "Potential business impact"
        }
    ],
    "overall_assessment": "2-3 sentence overall risk assessment",
    "risk_level": "high" | "moderate" | "low",
    "confidence": 0.0-1.0
}

Requirements:
- Include 3-5 of the most significant risks
- Order risks by severity (highest first)
- Be specific and avoid generic descriptions
- Focus on material risks that could affect business operations

JSON Response:"""


class StructuredSummarizer:
    """Generate structured JSON summaries from 10-K risk factors."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required for summarization")
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client
    
    def summarize(
        self,
        section_text: str,
        ticker: Optional[str] = None,
        max_risks: int = 5,
    ) -> StructuredSummary:
        """
        Generate a structured summary of risk factors.
        
        Args:
            section_text: Risk factors text from 10-K
            ticker: Company ticker symbol
            max_risks: Maximum number of risks to include
            
        Returns:
            StructuredSummary object with validated data
        """
        logger.info(f"Generating structured summary{f' for {ticker}' if ticker else ''}")
        
        prompt = STRUCTURED_PROMPT.replace("{{SECTION_TEXT}}", section_text[:8000])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial risk analyst. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1500,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            data = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {result_text}")
            raise ValueError(f"Invalid JSON from GPT: {e}")
        
        # Add metadata
        data["ticker"] = ticker
        data["model"] = self.model
        
        # Validate and return
        try:
            summary = StructuredSummary(**data)
            logger.info(f"Generated summary with {len(summary.risks)} risks")
            return summary
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise ValueError(f"Summary validation failed: {e}")
    
    def summarize_safe(
        self,
        section_text: str,
        ticker: Optional[str] = None,
        fallback: bool = True,
    ) -> Optional[StructuredSummary]:
        """
        Summarize with error handling and optional fallback.
        
        Returns None if generation fails and fallback is disabled.
        """
        try:
            return self.summarize(section_text, ticker)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            
            if fallback:
                # Return a minimal valid summary
                return StructuredSummary(
                    ticker=ticker,
                    risks=[
                        RiskItem(
                            title="Analysis Unavailable",
                            severity="medium",
                            category="Other",
                            description="Unable to generate detailed risk analysis. Please review source document.",
                            potential_impact="Unknown - manual review required",
                        )
                    ],
                    overall_assessment="Automated analysis could not be completed. Please review the source 10-K filing directly.",
                    risk_level="moderate",
                    confidence=0.1,
                )
            return None


def create_mock_summary(ticker: str = "DEMO") -> StructuredSummary:
    """Create a mock summary for testing/demo purposes."""
    return StructuredSummary(
        ticker=ticker,
        risks=[
            RiskItem(
                title="Cybersecurity Threats",
                severity="high",
                category="Cybersecurity",
                description="The company faces increasing sophisticated cyber attacks that could compromise customer data and disrupt operations.",
                potential_impact="Data breach could result in regulatory fines, litigation, and reputational damage.",
                source_paragraph=3,
            ),
            RiskItem(
                title="Supply Chain Disruption",
                severity="high",
                category="Supply Chain",
                description="Dependence on limited suppliers creates vulnerability to supply shortages and price increases.",
                potential_impact="Production delays could materially affect revenue and customer relationships.",
                source_paragraph=7,
            ),
            RiskItem(
                title="Regulatory Compliance",
                severity="medium",
                category="Regulatory",
                description="Evolving regulations across multiple jurisdictions require ongoing compliance investments.",
                potential_impact="Non-compliance could result in fines and operational restrictions.",
                source_paragraph=12,
            ),
        ],
        overall_assessment="The company faces elevated risk exposure across multiple domains, with cybersecurity and supply chain risks presenting the most significant near-term concerns. Management has disclosed appropriate mitigation strategies, but material adverse impact remains possible.",
        risk_level="high",
        confidence=0.85,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Structured risk summarization")
    parser.add_argument("--demo", action="store_true", help="Run demo with mock data")
    parser.add_argument("--file", type=str, help="Input file with risk text")
    parser.add_argument("--ticker", type=str, default="DEMO", help="Ticker symbol")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    if args.demo:
        summary = create_mock_summary(args.ticker)
        print("\n" + "="*60)
        print("STRUCTURED SUMMARY (DEMO)")
        print("="*60)
        print(summary.to_markdown())
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(summary.model_dump_json(indent=2))
            print(f"\n✅ Saved to {args.output}")
    
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        
        summarizer = StructuredSummarizer()
        summary = summarizer.summarize(text, ticker=args.ticker)
        
        print("\n" + "="*60)
        print("STRUCTURED SUMMARY")
        print("="*60)
        print(summary.to_markdown())
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(summary.model_dump_json(indent=2))
            print(f"\n✅ Saved to {args.output}")
    
    else:
        print("Usage: python structured_summarizer.py --demo")
        print("       python structured_summarizer.py --file <risks.txt> --ticker AAPL")

