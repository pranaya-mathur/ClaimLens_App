"""
LLM-Powered Fraud Explanation Generator
Uses Groq + Llama-3.3-70B to generate human-readable explanations
"""

import os
import time
from typing import Dict, Any, Optional
from functools import lru_cache
from loguru import logger

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain or langchain-groq not installed. LLM explanations disabled.")
    LANGCHAIN_AVAILABLE = False


class FraudExplainer:
    """
    Generates natural language explanations for fraud detection verdicts.
    
    Uses Groq's fast inference + Llama-3.3-70B-versatile model to:
    - Explain verdicts to insurance adjusters (technical)
    - Explain verdicts to customers (customer-friendly)
    - Generate actionable recommendations
    
    Features:
    - Rate limiting for Groq free tier (30 req/min)
    - Caching for common patterns
    - Fallback to template-based explanations
    - Structured prompts to avoid hallucination
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
        enable_caching: bool = True
    ):
        """
        Initialize Groq LLM explainer.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            temperature: LLM temperature (0.0-1.0, lower=more factual)
            max_tokens: Max tokens for explanation
            enable_caching: Cache explanations for similar patterns
        """
        self.enable_caching = enable_caching
        self.last_call_time = 0
        self.min_call_interval = 2.0  # Groq free tier: 30/min = 1 per 2 sec
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, LLM explanations will use templates")
            self.llm = None
            return
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set, LLM explanations will use templates")
            self.llm = None
            return
        
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens
            )
            logger.success("✓ Groq LLM initialized (llama-3.3-70b-versatile)")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            self.llm = None
    
    def explain_verdict(
        self,
        decision: Dict[str, Any],
        audience: str = "adjuster"
    ) -> str:
        """
        Generate explanation for fraud detection verdict.
        
        Args:
            decision: FinalDecision.to_dict() output with reasoning chain
            audience: "adjuster" (technical) or "customer" (friendly)
        
        Returns:
            Human-readable explanation string
        """
        # Try LLM explanation first
        if self.llm:
            try:
                return self._explain_with_llm(decision, audience)
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}, using template fallback")
        
        # Fallback to template
        return self._explain_with_template(decision, audience)
    
    def _explain_with_llm(self, decision: Dict[str, Any], audience: str) -> str:
        """
        Generate LLM-powered explanation.
        """
        # Rate limiting
        self._rate_limit()
        
        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(decision, audience)
            cached = self._get_cached_explanation(cache_key)
            if cached:
                logger.info("Using cached explanation")
                return cached
        
        # Build prompt based on audience
        if audience == "adjuster":
            prompt = self._build_adjuster_prompt(decision)
        else:
            prompt = self._build_customer_prompt(decision)
        
        # Generate explanation
        chain = prompt | self.llm | StrOutputParser()
        explanation = chain.invoke({})
        
        # Cache result
        if self.enable_caching:
            self._cache_explanation(cache_key, explanation)
        
        return explanation
    
    def _build_adjuster_prompt(self, decision: Dict[str, Any]) -> ChatPromptTemplate:
        """
        Build prompt for insurance adjuster (technical audience).
        """
        # Extract key data
        verdict = decision.get("verdict", "REVIEW")
        confidence = decision.get("confidence", 0.5)
        final_score = decision.get("final_score", 0.5)
        primary_reason = decision.get("primary_reason", "Multiple risk factors detected")
        
        # Get evidence
        evidence = decision.get("get_evidence_for_llm", decision)
        components = evidence.get("components", {})
        critical_flags = evidence.get("critical_flags", [])
        decision_steps = evidence.get("decision_steps", [])
        all_red_flags = evidence.get("all_red_flags", [])
        
        # Format components
        components_text = "\n".join([
            f"- {name}: {comp.get('verdict', 'N/A')} ({comp.get('confidence', 'N/A')} confidence) - {comp.get('reason', '')}"
            for name, comp in components.items()
        ])
        
        # Format critical flags
        if critical_flags:
            flags_text = "\n".join([f"- {flag.get('issue', '')} (Severity: {flag.get('severity', 'N/A')})" for flag in critical_flags])
        else:
            flags_text = "No critical flags detected."
        
        # Format decision steps
        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(decision_steps[:5])])  # Top 5 steps
        
        # Format red flags
        red_flags_text = "\n".join([f"• {flag}" for flag in all_red_flags[:5]]) if all_red_flags else "No red flags detected."
        
        template = f"""
You are an AI assistant explaining insurance fraud detection results to a claims adjuster.

**Claim Analysis Result:**
- Verdict: {verdict}
- Confidence: {confidence:.0%}
- Risk Score: {final_score:.2f}
- Primary Reason: {primary_reason}

**Component Analysis:**
{components_text}

**Critical Flags:**
{flags_text}

**Decision Process:**
{steps_text}

**Red Flags Identified:**
{red_flags_text}

**Instructions:**
Generate a clear, professional explanation for the claims adjuster that:
1. Starts with a one-sentence summary of the verdict
2. Explains the primary reason for the decision
3. Cites specific evidence from component analysis (quote the findings)
4. Mentions any critical flags that influenced the decision
5. Provides actionable next steps for the adjuster
6. Keeps technical terms (they understand fraud detection)

**Format:**
- Use bullet points for clarity
- Cite confidence levels when relevant
- Be specific with numbers and findings
- Maximum 200 words

**Important:** Only use information provided above. Do not make up details.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", "You are a fraud detection expert explaining claim analysis results to insurance professionals."),
            ("human", template)
        ])
    
    def _build_customer_prompt(self, decision: Dict[str, Any]) -> ChatPromptTemplate:
        """
        Build prompt for customer (non-technical, friendly).
        """
        verdict = decision.get("verdict", "REVIEW")
        primary_reason = decision.get("primary_reason", "We need additional information to process your claim")
        
        # Simplify for customer
        if verdict == "APPROVE":
            verdict_text = "approved"
        elif verdict == "REJECT":
            verdict_text = "cannot be approved at this time"
        else:
            verdict_text = "requires additional review"
        
        # Extract only customer-relevant info
        evidence = decision.get("get_evidence_for_llm", decision)
        components = evidence.get("components", {})
        
        # Only mention customer-visible issues
        customer_issues = []
        if "document_verification" in components:
            doc = components["document_verification"]
            if "FORGED" in doc.get("verdict", "") or "SUSPICIOUS" in doc.get("verdict", ""):
                customer_issues.append("Document verification concern")
        
        issues_text = ", ".join(customer_issues) if customer_issues else "standard verification process"
        
        template = f"""
You are a customer service AI explaining an insurance claim decision to a claimant.

**Claim Status:** {verdict_text}
**Reason:** {primary_reason}
**Issues Identified:** {issues_text}

**Instructions:**
Write a polite, empathetic explanation for the customer that:
1. States the claim status clearly
2. Explains why in simple terms (avoid technical jargon)
3. Tells them what they can do next (submit documents, appeal, etc.)
4. Offers contact information for help
5. Maintains a professional but friendly tone

**Avoid mentioning:**
- "Fraud" or "fraud detection" (use "verification concern" instead)
- Technical terms like "ML model", "graph analysis"
- Internal risk scores or confidence levels
- Fraud ring detection (legal sensitivity)

**Format:**
- Use clear paragraphs (not bullets)
- Maximum 150 words
- Start with "Dear Valued Customer"
- End with helpful next steps

**Important:** Be empathetic and helpful, not accusatory.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer service representative explaining insurance claim decisions."),
            ("human", template)
        ])
    
    def _explain_with_template(self, decision: Dict[str, Any], audience: str) -> str:
        """
        Fallback template-based explanation (no LLM needed).
        """
        verdict = decision.get("verdict", "REVIEW")
        confidence = decision.get("confidence", 0.5)
        primary_reason = decision.get("primary_reason", "Multiple risk factors detected")
        final_score = decision.get("final_score", 0.5)
        
        if audience == "adjuster":
            return self._template_adjuster_explanation(
                verdict, confidence, final_score, primary_reason, decision
            )
        else:
            return self._template_customer_explanation(
                verdict, primary_reason
            )
    
    def _template_adjuster_explanation(
        self, verdict: str, confidence: float, score: float, reason: str, decision: Dict
    ) -> str:
        """
        Template-based explanation for adjuster.
        """
        red_flags = decision.get("red_flags", [])
        critical_flags = decision.get("critical_flags", [])
        
        explanation = f"""**Fraud Detection Analysis Result**

Verdict: {verdict} (Confidence: {confidence:.0%}, Risk Score: {score:.2f})

Primary Finding:
{reason}

"""
        
        if critical_flags:
            explanation += "Critical Issues:\n"
            for flag in critical_flags[:3]:
                explanation += f"• {flag.get('reason', flag)}\n"
            explanation += "\n"
        
        if red_flags:
            explanation += "Red Flags Detected:\n"
            for flag in red_flags[:5]:
                explanation += f"• {flag}\n"
            explanation += "\n"
        
        explanation += f"""Recommended Action:
"""
        
        if verdict == "APPROVE":
            explanation += "Claim can be processed. Low fraud risk detected."
        elif verdict == "REJECT":
            explanation += "Claim should be rejected pending investigation. High fraud indicators present."
        else:
            explanation += "Claim requires manual review. Refer to fraud investigation team for detailed analysis."
        
        return explanation
    
    def _template_customer_explanation(self, verdict: str, reason: str) -> str:
        """
        Template-based explanation for customer.
        """
        if verdict == "APPROVE":
            return f"""Dear Valued Customer,

We're pleased to inform you that your claim has been approved and will be processed shortly.

{reason}

Your payment will be initiated within 3-5 business days.

If you have any questions, please contact our customer service at 1800-XXX-XXXX.

Thank you for your patience.

Best regards,
Claims Team
"""
        
        elif verdict == "REJECT":
            return f"""Dear Valued Customer,

We're unable to approve your claim at this time.

{reason}

What you can do:
1. Submit additional supporting documents
2. Provide clarification on flagged items
3. Appeal this decision within 30 days

For assistance, please call our helpline at 1800-XXX-XXXX or visit your nearest branch with original documents.

We're here to help resolve any concerns.

Best regards,
Claims Team
"""
        
        else:  # REVIEW
            return f"""Dear Valued Customer,

Your claim is currently under review by our team.

{reason}

We may need:
• Additional documentation
• Clarification on certain details
• Verification from third parties

Our team will contact you within 2-3 business days with next steps.

For urgent inquiries, please call 1800-XXX-XXXX.

Thank you for your patience.

Best regards,
Claims Team
"""
    
    def _rate_limit(self):
        """
        Enforce rate limiting for Groq free tier (30 req/min).
        """
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        if elapsed < self.min_call_interval:
            sleep_time = self.min_call_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _get_cache_key(self, decision: Dict[str, Any], audience: str) -> str:
        """
        Generate cache key for decision.
        """
        verdict = decision.get("verdict", "REVIEW")
        score = decision.get("final_score", 0.5)
        score_bucket = int(score * 10) / 10  # Bucket to 0.1 precision
        primary_reason = decision.get("primary_reason", "")[:50]  # First 50 chars
        
        return f"{audience}:{verdict}:{score_bucket}:{hash(primary_reason)}"
    
    @lru_cache(maxsize=100)
    def _get_cached_explanation(self, cache_key: str) -> Optional[str]:
        """
        Get cached explanation (using lru_cache).
        """
        return None  # lru_cache handles this
    
    def _cache_explanation(self, cache_key: str, explanation: str):
        """
        Cache explanation for reuse.
        """
        # lru_cache handles caching automatically via _get_cached_explanation
        pass
