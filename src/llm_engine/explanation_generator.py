"""
Explanation Generator using Groq LLM

Generates human-friendly explanations for fraud detection verdicts.
Supports both technical (adjuster) and friendly (customer) audiences.
Includes streaming mode for real-time explanation display.
"""

import os
from typing import Dict, Any, Optional, Generator
from loguru import logger
from groq import Groq


class ExplanationGenerator:
    """
    LLM-powered explanation generator for fraud detection results.
    
    Generates clear, audience-appropriate explanations:
    - Adjuster Mode: Technical details, risk factors, next steps
    - Customer Mode: Friendly language, clear reasoning, empathy
    
    Features:
    - Streaming support for real-time display
    - Multi-language support (English, Hinglish)
    - Contextual explanations based on verdict
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        Initialize Explanation Generator.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Groq model to use
            temperature: LLM temperature (0.3 for balanced creativity)
            max_tokens: Max tokens for explanation
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY not found. Explanations will use templates. "
                "Get your free API key from https://console.groq.com/"
            )
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.success(f"Explanation generator initialized with {model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _build_explanation_prompt(
        self,
        verdict_data: Dict[str, Any],
        claim_data: Dict[str, Any],
        component_results: Dict[str, Any],
        audience: str = "adjuster"
    ) -> str:
        """
        Build prompt for generating explanation.
        
        Args:
            verdict_data: Final verdict from semantic aggregation
            claim_data: Original claim information
            component_results: Individual component results
            audience: Target audience ("adjuster" or "customer")
            
        Returns:
            Structured prompt for LLM
        """
        audience_lower = audience.lower()
        
        if "customer" in audience_lower:
            audience_type = "customer"
            tone = "friendly, empathetic, and clear"
            technical_level = "minimal"
        else:
            audience_type = "insurance adjuster"
            tone = "professional and technical"
            technical_level = "detailed"
        
        prompt = f"""
You are explaining an insurance claim fraud detection decision to a {audience_type}.

# CLAIM DETAILS
Claim ID: {claim_data.get('claim_id', 'N/A')}
Product: {claim_data.get('product', 'N/A')}
Claim Amount: ₹{claim_data.get('claim_amount', 0):,}
Premium: ₹{claim_data.get('premium', 0):,}
Policy Age: {claim_data.get('days_since_policy', 0)} days
Narrative: {claim_data.get('narrative', 'N/A')}

# AI ANALYSIS RESULT
Final Verdict: {verdict_data.get('verdict', 'UNKNOWN')}
Confidence: {verdict_data.get('confidence', 0):.0%}
Risk Score: {verdict_data.get('final_risk_score', 0):.2f}
Primary Reason: {verdict_data.get('primary_reason', 'N/A')}
Critical Flags: {', '.join(verdict_data.get('critical_flags', [])) or 'None'}

# COMPONENT ANALYSIS
"""
        
        for comp_name, result in component_results.items():
            prompt += f"""
{comp_name.replace('_', ' ').title()}:
- Status: {result.get('verdict', 'N/A')}
- Confidence: {result.get('confidence', 0):.0%}
- Details: {result.get('reason', 'N/A')}
"""
        
        prompt += f"""

# YOUR TASK
Write a clear, {tone} explanation (3-4 paragraphs, ~150-200 words) for the {audience_type} that:

"""
        
        if audience_type == "customer":
            prompt += """
1. Explains the decision in simple terms (avoid jargon)
2. Describes what was checked (documents, history, patterns)
3. States the outcome clearly and what happens next
4. Shows empathy and offers next steps if needed
5. Uses friendly language ("we", "your claim", etc.)

DO NOT:
- Use technical terms like "ML model", "fraud probability", "risk score"
- Make the customer feel accused or defensive
- Include complex statistics or percentages

Write in a warm, helpful tone as if you're speaking to a friend.
"""
        else:
            prompt += f"""
1. Summarize the verdict and confidence level
2. Explain key risk factors from each component
3. Highlight critical flags or red flags
4. Provide recommended next steps
5. Include {technical_level} technical details

DO:
- Use precise terminology (fraud probability, risk scores)
- Reference specific component verdicts
- Quantify risks where relevant
- Suggest follow-up actions

Write in a professional, analytical tone.
"""
        
        prompt += """

Respond with ONLY the explanation text (no JSON, no headers, no formatting).
"""
        
        return prompt
    
    def _generate_template_explanation(
        self,
        verdict_data: Dict[str, Any],
        claim_data: Dict[str, Any],
        component_results: Dict[str, Any],
        audience: str = "adjuster"
    ) -> str:
        """
        Generate template-based explanation when LLM unavailable.
        
        Args:
            verdict_data: Final verdict
            claim_data: Claim information
            component_results: Component results
            audience: Target audience
            
        Returns:
            Template explanation string
        """
        logger.info("Using template explanation (no LLM)")
        
        verdict = verdict_data.get('verdict', 'REVIEW')
        confidence = verdict_data.get('confidence', 0)
        risk_score = verdict_data.get('final_risk_score', 0)
        claim_amount = claim_data.get('claim_amount', 0)
        premium = claim_data.get('premium', 1)
        days = claim_data.get('days_since_policy', 0)
        ratio = claim_amount / max(premium, 1)
        
        if "customer" in audience.lower():
            # Customer-friendly explanation
            if verdict == "APPROVE":
                explanation = f"""Good news! After carefully reviewing your claim, we've determined it can be processed for approval. 

We checked your documents, claim history, and other important details. Everything appears to be in order with no significant concerns. Your claim for ₹{claim_amount:,} has been verified against your policy details.

Next steps: Your claim will move forward to our payment processing team. You should receive an update on payment within 3-5 business days.

If you have any questions, please don't hesitate to contact our customer service team. We're here to help!"""
            
            elif verdict == "REVIEW":
                explanation = f"""Thank you for submitting your claim. After our initial review, we need to look into a few more details before making a final decision.

We've reviewed your documents and claim information. While most things look good, we noticed some details that require additional verification to ensure everything is accurate. This is a standard part of our review process.

Next steps: One of our claims adjusters will contact you within 2-3 business days to discuss the claim further. Please keep any additional documentation ready that might support your claim.

We appreciate your patience as we work to resolve this fairly and accurately."""
            
            else:  # REJECT
                explanation = f"""After carefully reviewing your claim, we're unable to approve it at this time.

Our review process checks several factors including document authenticity, claim patterns, and policy details. Unfortunately, we found concerns that prevent us from moving forward with this claim.

Next steps: You have the right to appeal this decision. Please contact our claims department within 30 days to discuss your options. We'll need any additional information or documentation that might address the concerns we identified.

We understand this may be disappointing, and we're available to discuss this decision with you in more detail."""
        
        else:
            # Adjuster/technical explanation
            ml_verdict = component_results.get('ml_fraud_score', {}).get('verdict', 'UNKNOWN')
            ml_prob = component_results.get('ml_fraud_score', {}).get('score', 0) * 100
            doc_verdict = component_results.get('document_verification', {}).get('verdict', 'UNKNOWN')
            graph_verdict = component_results.get('graph_analysis', {}).get('verdict', 'UNKNOWN')
            
            explanation = f"""Verdict: {verdict} (Confidence: {confidence:.0%}, Risk Score: {risk_score:.2f})

This claim requires {verdict.lower()} based on multi-component analysis. The claim amount of ₹{claim_amount:,} against premium ₹{premium:,} ({ratio:.1f}x ratio) was filed {days} days after policy inception.

Component Analysis:
- ML Fraud Score: {ml_verdict} ({ml_prob:.0f}% fraud probability)
- Document Verification: {doc_verdict}
- Graph Analysis: {graph_verdict}
"""
            
            critical_flags = verdict_data.get('critical_flags', [])
            if critical_flags:
                explanation += f"\nCritical Flags: {', '.join(critical_flags)}"
            
            if verdict == "APPROVE":
                explanation += "\n\nRecommendation: Process claim with standard verification. No additional investigation required."
            elif verdict == "REVIEW":
                explanation += "\n\nRecommendation: Conduct manual review. Verify claimant history, document authenticity, and claim narrative details."
            else:
                explanation += "\n\nRecommendation: Deny claim and flag claimant for monitoring. Consider fraud investigation if patterns emerge."
        
        return explanation
    
    def generate(
        self,
        verdict_data: Dict[str, Any],
        claim_data: Dict[str, Any],
        component_results: Dict[str, Any],
        audience: str = "adjuster",
        stream: bool = False
    ) -> str:
        """
        Generate explanation for fraud detection verdict.
        
        Args:
            verdict_data: Final verdict from semantic aggregation
            claim_data: Original claim information
            component_results: Individual component results
            audience: Target audience ("adjuster" or "customer")
            stream: Whether to return generator for streaming (not yet implemented)
            
        Returns:
            Explanation text
        """
        logger.info(f"Generating {audience} explanation for {claim_data.get('claim_id')}")
        
        # Fallback if LLM unavailable
        if not self.client:
            return self._generate_template_explanation(
                verdict_data, claim_data, component_results, audience
            )
        
        try:
            # Build prompt
            prompt = self._build_explanation_prompt(
                verdict_data, claim_data, component_results, audience
            )
            
            # Call Groq API
            logger.info(f"Generating explanation with {self.model}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful insurance claims expert explaining decisions to {audience}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            explanation = response.choices[0].message.content.strip()
            
            logger.success(
                f"Explanation generated: {len(explanation)} chars, "
                f"{response.usage.total_tokens} tokens"
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return self._generate_template_explanation(
                verdict_data, claim_data, component_results, audience
            )
    
    def generate_streaming(
        self,
        verdict_data: Dict[str, Any],
        claim_data: Dict[str, Any],
        component_results: Dict[str, Any],
        audience: str = "adjuster"
    ) -> Generator[str, None, None]:
        """
        Generate explanation with streaming support.
        
        Args:
            verdict_data: Final verdict
            claim_data: Claim information
            component_results: Component results
            audience: Target audience
            
        Yields:
            Chunks of explanation text as they're generated
        """
        logger.info(f"Generating streaming {audience} explanation")
        
        # Fallback if LLM unavailable
        if not self.client:
            explanation = self._generate_template_explanation(
                verdict_data, claim_data, component_results, audience
            )
            # Simulate streaming by yielding words
            for word in explanation.split():
                yield word + " "
            return
        
        try:
            # Build prompt
            prompt = self._build_explanation_prompt(
                verdict_data, claim_data, component_results, audience
            )
            
            # Call Groq API with streaming
            logger.info(f"Streaming explanation with {self.model}...")
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful insurance claims expert explaining decisions to {audience}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            logger.success("Streaming explanation complete")
            
        except Exception as e:
            logger.error(f"Streaming explanation error: {e}")
            # Fallback to template
            explanation = self._generate_template_explanation(
                verdict_data, claim_data, component_results, audience
            )
            for word in explanation.split():
                yield word + " "
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            True if Groq client is initialized
        """
        return self.client is not None
