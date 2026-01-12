"""
Phase 2.5: Production-Hardened GPT Summarizer

Implements robust summarization with:
- Retry logic with exponential backoff
- Request timeouts
- Fallback strategies
- Rate limit handling
- Error classification and logging
- Token counting and cost tracking

This is the production-ready version of the summarizer.
"""

import os
import sys
import json
import time
import logging
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, get_logger

logger = get_logger(__name__)

# === Configuration ===

@dataclass
class SummarizerConfig:
    """Configuration for production summarizer."""
    model: str = "gpt-4o"
    temperature: float = 0.2  # Recommended from Phase 2.4
    max_tokens: int = 500
    
    # Retry settings
    max_retries: int = 3
    initial_retry_delay: float = 1.0  # seconds
    max_retry_delay: float = 30.0  # seconds
    retry_multiplier: float = 2.0
    
    # Timeout settings
    request_timeout: float = 60.0  # seconds
    
    # Rate limiting
    requests_per_minute: int = 50
    
    # Cost tracking (approximate)
    cost_per_1k_input_tokens: float = 0.005  # GPT-4o pricing
    cost_per_1k_output_tokens: float = 0.015


# === Error Types ===

class SummarizerError(Exception):
    """Base exception for summarizer errors."""
    pass


class RetryableError(SummarizerError):
    """Error that should be retried."""
    pass


class NonRetryableError(SummarizerError):
    """Error that should not be retried."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded."""
    pass


class TimeoutError(RetryableError):
    """Request timed out."""
    pass


class InvalidResponseError(NonRetryableError):
    """Invalid response from API."""
    pass


class AuthenticationError(NonRetryableError):
    """Authentication failed."""
    pass


# === Metrics Tracking ===

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    error_type: Optional[str] = None
    model: str = ""


@dataclass
class AggregateMetrics:
    """Aggregated metrics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_request(self, metrics: RequestMetrics, config: SummarizerConfig):
        """Add a request's metrics to aggregate."""
        self.total_requests += 1
        self.total_retries += metrics.retries
        
        if metrics.success:
            self.successful_requests += 1
            self.total_input_tokens += metrics.input_tokens
            self.total_output_tokens += metrics.output_tokens
            
            # Calculate cost
            input_cost = (metrics.input_tokens / 1000) * config.cost_per_1k_input_tokens
            output_cost = (metrics.output_tokens / 1000) * config.cost_per_1k_output_tokens
            self.total_cost += input_cost + output_cost
            
            # Update average latency
            total_latency = self.avg_latency_ms * (self.successful_requests - 1)
            self.avg_latency_ms = (total_latency + metrics.latency_ms) / self.successful_requests
        else:
            self.failed_requests += 1
            error = metrics.error_type or "unknown"
            self.error_counts[error] = self.error_counts.get(error, 0) + 1
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{self.success_rate:.1%}",
            "total_retries": self.total_retries,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": f"${self.total_cost:.4f}",
            "avg_latency_ms": int(self.avg_latency_ms),
            "error_counts": self.error_counts,
        }


# === Retry Decorator ===

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    multiplier: float = 2.0,
    retryable_exceptions: tuple = (RetryableError,),
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        multiplier: Delay multiplier after each retry
        retryable_exceptions: Exception types to retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Add jitter to prevent thundering herd
                        jitter = random.uniform(0.5, 1.5)
                        sleep_time = min(delay * jitter, max_delay)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= multiplier
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise
                except NonRetryableError:
                    # Don't retry these
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator


# === Production Summarizer ===

class ProductionSummarizer:
    """
    Production-ready GPT summarizer with robust error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Request timeout handling
    - Rate limit management
    - Fallback strategies
    - Metrics tracking
    """
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()
        self.metrics = AggregateMetrics()
        self._client = None
        self._last_request_time = 0
        self._min_request_interval = 60.0 / self.config.requests_per_minute
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise AuthenticationError("OPENAI_API_KEY required")
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=self.config.request_timeout,
            )
        return self._client
    
    def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _classify_error(self, error: Exception) -> Exception:
        """Classify OpenAI errors into retryable/non-retryable."""
        error_str = str(error).lower()
        
        if "rate_limit" in error_str or "429" in str(type(error)):
            return RateLimitError(str(error))
        elif "timeout" in error_str or "timed out" in error_str:
            return TimeoutError(str(error))
        elif "authentication" in error_str or "401" in str(type(error)):
            return AuthenticationError(str(error))
        elif "invalid" in error_str:
            return InvalidResponseError(str(error))
        else:
            # Default to retryable for transient errors
            return RetryableError(str(error))
    
    @with_retry(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def _call_api(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
    ) -> Dict:
        """
        Make API call with retry logic.
        
        Returns dict with response data.
        """
        self._rate_limit_wait()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return {
                "content": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }
        except Exception as e:
            raise self._classify_error(e)
    
    def summarize(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        fallback_text: Optional[str] = None,
    ) -> Dict:
        """
        Generate summary with full production hardening.
        
        Args:
            text: Text to summarize
            prompt_template: Custom prompt (use {{SECTION_TEXT}} placeholder)
            system_prompt: Custom system prompt
            fallback_text: Text to return if all attempts fail
        
        Returns:
            Dict with summary, metrics, and status
        """
        request_metrics = RequestMetrics(model=self.config.model)
        start_time = time.time()
        
        # Default prompts
        if prompt_template is None:
            prompt_template = """Summarize the following risk factors into 3-5 key points:

{{SECTION_TEXT}}

Summary:"""
        
        if system_prompt is None:
            system_prompt = "You are a professional financial analyst."
        
        # Build messages
        full_prompt = prompt_template.replace("{{SECTION_TEXT}}", text[:8000])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
        
        try:
            response = self._call_api(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            request_metrics.success = True
            request_metrics.input_tokens = response["input_tokens"]
            request_metrics.output_tokens = response["output_tokens"]
            request_metrics.latency_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "success": True,
                "summary": response["content"].strip(),
                "tokens_used": response["input_tokens"] + response["output_tokens"],
                "latency_ms": request_metrics.latency_ms,
            }
            
        except NonRetryableError as e:
            request_metrics.error_type = type(e).__name__
            logger.error(f"Non-retryable error: {e}")
            
            result = self._create_fallback_result(
                str(e), 
                fallback_text,
                int((time.time() - start_time) * 1000),
            )
            
        except RetryableError as e:
            request_metrics.error_type = type(e).__name__
            request_metrics.retries = self.config.max_retries
            logger.error(f"All retries exhausted: {e}")
            
            result = self._create_fallback_result(
                str(e),
                fallback_text,
                int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            request_metrics.error_type = "UnexpectedError"
            logger.exception(f"Unexpected error: {e}")
            
            result = self._create_fallback_result(
                str(e),
                fallback_text,
                int((time.time() - start_time) * 1000),
            )
        
        # Update aggregate metrics
        self.metrics.add_request(request_metrics, self.config)
        
        return result
    
    def _create_fallback_result(
        self,
        error: str,
        fallback_text: Optional[str],
        latency_ms: int,
    ) -> Dict:
        """Create fallback result when summarization fails."""
        if fallback_text:
            summary = fallback_text
        else:
            summary = (
                "Unable to generate summary due to a technical issue. "
                "Please review the source document directly."
            )
        
        return {
            "success": False,
            "summary": summary,
            "error": error,
            "latency_ms": latency_ms,
            "fallback": True,
        }
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = AggregateMetrics()


# === Factory Function ===

_default_summarizer: Optional[ProductionSummarizer] = None


def get_summarizer(config: Optional[SummarizerConfig] = None) -> ProductionSummarizer:
    """
    Get the production summarizer instance.
    
    Uses singleton pattern for efficiency.
    """
    global _default_summarizer
    
    if config is not None:
        return ProductionSummarizer(config)
    
    if _default_summarizer is None:
        _default_summarizer = ProductionSummarizer()
    
    return _default_summarizer


def summarize_section(
    section_text: str,
    section_name: str = "Risk Factors",
    **kwargs,
) -> str:
    """
    High-level API for summarizing a section.
    
    Drop-in replacement for the original summarizer.
    """
    summarizer = get_summarizer()
    result = summarizer.summarize(section_text, **kwargs)
    
    if result["success"]:
        logger.info(f"✅ Summary generated for {section_name}")
    else:
        logger.warning(f"⚠️ Fallback summary used for {section_name}")
    
    return result["summary"]


# === Testing ===

def test_production_summarizer():
    """Test the production summarizer."""
    print("\n" + "="*60)
    print("PRODUCTION SUMMARIZER TEST")
    print("="*60)
    
    # Create summarizer with test config
    config = SummarizerConfig(
        max_retries=2,
        initial_retry_delay=0.5,
    )
    
    summarizer = ProductionSummarizer(config)
    
    test_text = """
    The Company faces significant cybersecurity threats that could result in 
    data breaches, unauthorized access to confidential information, and disruption 
    to business operations. Supply chain disruptions could materially affect 
    manufacturing capacity and product availability. Regulatory compliance 
    requirements continue to evolve, creating ongoing legal and operational risks.
    """
    
    print("\n📝 Test input:", test_text[:100] + "...")
    
    result = summarizer.summarize(test_text)
    
    print(f"\n✅ Success: {result['success']}")
    print(f"📊 Latency: {result.get('latency_ms', 'N/A')}ms")
    print(f"\n📄 Summary:\n{result['summary']}")
    
    if result.get('error'):
        print(f"\n⚠️ Error: {result['error']}")
    
    print("\n📈 Metrics:")
    print(json.dumps(summarizer.get_metrics(), indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production summarizer")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--file", type=str, help="File to summarize")
    args = parser.parse_args()
    
    if args.test:
        test_production_summarizer()
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        
        result = get_summarizer().summarize(text)
        print(result["summary"])
    else:
        print("Usage: python production_summarizer.py --test")
        print("       python production_summarizer.py --file <input.txt>")

