"""
Semantic Aggregator using LangChain + Groq LLM (Llama-3.3-70B)

Performs intelligent verdict synthesis by analyzing multiple fraud detection
components (ML, documents, graph) and generating a final semantic verdict.
"""

import os
import json
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("langchain-groq not installed. Install with: pip install langchain-groq")
    LANGCHAIN_AVAILABLE = False


class SemanticAggregator:
    """
    LLM-powered semantic aggregation for fraud detection verdicts.
    
    Uses Groq's Llama-3.3-70B via LangChain to intelligently combine results from:
    - Document verification (PAN, Aadhaar, etc.)
    - ML fraud scoring
    - Graph network analysis
    
    Generates:
    - Final verdict (APPROVE/REVIEW/REJECT)
    - Confidence score
    - Reasoning chain
    - Critical flags
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize Semantic Aggregator.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Groq model to use (default: llama-3.3-70b-versatile)
            temperature: LLM temperature (0.1 for consistent analysis)
            max_tokens: Max tokens for response
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available. Semantic aggregation will use fallback logic.")
            return
        
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY not found. Semantic aggregation will use fallback logic. "
                "Get your free API key from https://console.groq.com/"
            )
        else:
            try:
                self.llm = ChatGroq(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.success(f"LangChain ChatGroq initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq: {e}")
                self.llm = None
    
    def _build_analysis_prompt(self, component_results: Dict[str, Any], claim_data: Dict[str, Any]) -> str:
        """
        Build comprehensive prompt for LLM analysis.
        
        Args:
            component_results: Results from each detection component
            claim_data: Original claim information
            
        Returns:
            Structured prompt for LLM
        """
        prompt = f"""
You are an expert insurance fraud analyst. Analyze the following claim and provide a final verdict.

# CLAIM INFORMATION
Claim ID: {claim_data.get('claim_id', 'N/A')}
Product: {claim_data.get('product', 'N/A')}
Subtype: {claim_data.get('subtype', 'N/A')}
Claim Amount: ₹{claim_data.get('claim_amount', 0):,}
Premium: ₹{claim_data.get('premium', 0):,}
Claim-to-Premium Ratio: {claim_data.get('claim_amount', 0) / max(claim_data.get('premium', 1), 1):.1f}x
Days Since Policy: {claim_data.get('days_since_policy', 0)} days
Narrative: {claim_data.get('narrative', 'N/A')}

# COMPONENT ANALYSIS RESULTS
"""
        
        for comp_name, result in component_results.items():
            prompt += f"""
## {comp_name.replace('_', ' ').upper()}
- Verdict: {result.get('verdict', 'UNKNOWN')}
- Confidence: {result.get('confidence', 0):.0%}
- Risk Score: {result.get('score', 0):.2f}
- Reason: {result.get('reason', 'N/A')}
- Red Flags: {', '.join(result.get('red_flags', [])) or 'None'}
"""
        
        prompt += """

# YOUR TASK
Based on the above analysis, provide:

1. **FINAL_VERDICT**: One of [APPROVE, REVIEW, REJECT]
2. **CONFIDENCE**: Float between 0-1 indicating certainty
3. **FINAL_RISK_SCORE**: Float between 0-1 (0=no fraud, 1=definite fraud)
4. **PRIMARY_REASON**: One-sentence main justification
5. **CRITICAL_FLAGS**: List of major concerns (if any)
6. **REASONING_CHAIN**: Step-by-step decision logic

IMPORTANT GUIDELINES:
- APPROVE: Low fraud indicators, standard processing (risk < 0.3)
- REVIEW: Mixed signals, needs manual review (risk 0.3-0.7)
- REJECT: High fraud probability, deny claim (risk > 0.7)
- Weight document forgery heavily (authentic documents reduce risk)
- Consider claim-to-premium ratio (>10x is suspicious)
- Recent policies (<90 days) with large claims are risky
- Fraud network connections are critical red flags
- Balance severity vs confidence in components

Respond ONLY with valid JSON (no markdown, no explanations outside JSON):
{
  "verdict": "APPROVE|REVIEW|REJECT",
  "confidence": 0.85,
  "final_risk_score": 0.42,
  "primary_reason": "Brief explanation",
  "critical_flags": ["flag1", "flag2"],
  "reasoning_chain": [
    {"stage": "stage_name", "decision": "decision", "reason": "why"}
  ],
  "recommendation": "Action to take"
}
"""
        
        return prompt
    
    def _fallback_aggregation(self, component_results: Dict[str, Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback logic when LLM is unavailable.
        
        Uses weighted averaging and rule-based decision.
        """
        logger.info("Using fallback aggregation (no LLM)")
        
        # Weighted scoring
        doc_weight = 0.3
        ml_weight = 0.4
        graph_weight = 0.3
        
        doc_score = component_results.get('document_verification', {}).get('score', 0) * doc_weight
        ml_score = component_results.get('ml_fraud_score', {}).get('score', 0) * ml_weight
        graph_score = component_results.get('graph_analysis', {}).get('score', 0) * graph_weight
        
        final_score = doc_score + ml_score + graph_score
        
        # Determine verdict
        if final_score >= 0.7:
            verdict = "REJECT"
            primary_reason = "High fraud probability detected across multiple components"
        elif final_score >= 0.3:
            verdict = "REVIEW"
            ratio = claim_data.get('claim_amount', 0) / max(claim_data.get('premium', 1), 1)
            days = claim_data.get('days_since_policy', 0)
            primary_reason = f"Claim amount ₹{claim_data.get('claim_amount', 0):,} against premium ₹{claim_data.get('premium', 0):,} ({ratio:.0f}x ratio) with {days} days policy age requires review"
        else:
            verdict = "APPROVE"
            primary_reason = "Low fraud indicators detected, claim appears legitimate"
        
        confidence = 1 - abs(final_score - 0.5) * 2
        
        # Critical flags
        critical_flags = []
        
        if component_results.get('document_verification', {}).get('verdict') in ['FORGED', 'SUSPICIOUS']:
            critical_flags.append("Document authenticity concerns detected")
        
        if component_results.get('graph_analysis', {}).get('verdict') == 'FRAUD_RING_DETECTED':
            critical_flags.append("Fraud network connections identified")
        
        ratio = claim_data.get('claim_amount', 0) / max(claim_data.get('premium', 1), 1)
        if ratio > 10:
            critical_flags.append(f"Extremely high claim-to-premium ratio: {ratio:.0f}x")
        
        if claim_data.get('days_since_policy', 999) < 90:
            critical_flags.append(f"Recent policy: {claim_data.get('days_since_policy')} days old")
        
        # Reasoning chain
        reasoning_chain = [
            {
                "stage": "document_verification",
                "decision": component_results.get('document_verification', {}).get('verdict', 'N/A'),
                "reason": "Document authenticity analysis completed"
            },
            {
                "stage": "ml_fraud_scoring",
                "decision": component_results.get('ml_fraud_score', {}).get('verdict', 'N/A'),
                "reason": "Machine learning risk assessment completed"
            },
            {
                "stage": "graph_analysis",
                "decision": component_results.get('graph_analysis', {}).get('verdict', 'N/A'),
                "reason": "Network fraud analysis completed"
            },
            {
                "stage": "final_verdict",
                "decision": verdict,
                "reason": f"Aggregated risk score: {final_score:.2f}"
            }
        ]
        
        recommendation = f"{verdict} - {primary_reason}"
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "final_risk_score": final_score,
            "primary_reason": primary_reason,
            "critical_flags": critical_flags,
            "reasoning_chain": reasoning_chain,
            "recommendation": recommendation,
            "llm_used": False
        }
    
    def aggregate(
        self,
        component_results: Dict[str, Any],
        claim_data: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Perform semantic aggregation of fraud detection results.
        
        Args:
            component_results: Results from ML, docs, graph components
            claim_data: Original claim information
            stream: Whether to stream response (not implemented in aggregation)
            
        Returns:
            Dictionary with final verdict and reasoning
        """
        logger.info(f"Starting semantic aggregation for claim {claim_data.get('claim_id')}")
        
        # Fallback if LLM unavailable
        if not self.llm:
            return self._fallback_aggregation(component_results, claim_data)
        
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(component_results, claim_data)
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Call LangChain ChatGroq
            logger.info(f"Calling ChatGroq {self.model}...")
            
            messages = [
                SystemMessage(content="You are an expert insurance fraud analyst. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse response
            result = json.loads(content)
            
            # Validate result
            if 'verdict' not in result:
                logger.error("LLM response missing 'verdict' field")
                return self._fallback_aggregation(component_results, claim_data)
            
            # Add metadata
            result['llm_used'] = True
            result['model'] = self.model
            
            # Get token usage if available
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                result['tokens_used'] = usage.get('total_tokens', 0)
            
            logger.success(
                f"Semantic aggregation complete: {result['verdict']} "
                f"(confidence: {result['confidence']:.0%})"
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_aggregation(component_results, claim_data)
        
        except Exception as e:
            logger.error(f"Semantic aggregation error: {e}")
            return self._fallback_aggregation(component_results, claim_data)
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            True if ChatGroq is initialized
        """
        return self.llm is not None
