# Semantic Aggregation & LLM Explainability Setup Guide

## 🎯 What's New in ClaimLens v2.0

ClaimLens now includes **Explainable AI** features:

### 1. **Semantic Verdicts**
- Instead of just numeric scores (0.73), get human-readable verdicts:
  - `FORGED`, `AUTHENTIC`, `SUSPICIOUS` (documents)
  - `HIGH_RISK`, `MEDIUM_RISK`, `LOW_RISK` (fraud scoring)
  - `FRAUD_RING_DETECTED`, `CLEAN` (graph analysis)
  - `APPROVE`, `REVIEW`, `REJECT` (final verdict)

### 2. **Critical Flag Gating**
- Hard constraints that override normal scoring:
  - High-confidence document forgery → Auto-reject
  - Fraud ring membership → Mandatory review
  - Excessive claim amounts → Flag for investigation

### 3. **Adaptive Weighting**
- Weights adjust based on:
  - Component confidence levels
  - Product type (motor/health/life)
  - Data availability (fallback usage)

### 4. **Full Reasoning Chain**
- Every decision step is logged:
  - Stage 1: Critical flag check
  - Stage 2: Adaptive scoring
  - Stage 3: Verdict determination
  - Stage 4: Data quality check

### 5. **LLM-Powered Explanations**
- Natural language explanations using Groq + Llama-3.3-70B:
  - **Adjuster mode**: Technical, detailed, cites evidence
  - **Customer mode**: Friendly, empathetic, actionable

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install latest LangChain for Groq integration
pip install langchain langchain-groq langchain-core langchain-community

# Or install all dependencies
pip install -r requirements.txt
```

### Step 2: Get Groq API Key

1. Go to https://console.groq.com/
2. Sign up for free account
3. Generate API key
4. Copy the key (starts with `gsk_...`)

**Groq Free Tier:**
- 30 requests per minute
- 14,400 tokens per minute
- Zero cost
- Super fast inference (~500 tokens/sec)

### Step 3: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=gsk_your_actual_key_here

# Enable features (default: true)
ENABLE_SEMANTIC_AGGREGATION=true
ENABLE_LLM_EXPLANATIONS=true
```

### Step 4: Test the System

```bash
# Run the demo
python examples/semantic_fraud_detection_demo.py
```

You should see:
- ✅ Semantic verdicts for each component
- ✅ Critical flags detected
- ✅ Full reasoning chain
- ✅ LLM-generated explanations

---

## 💻 Usage Examples

### Basic Usage (Semantic Mode)

```python
from src.app.claim_processor import ClaimProcessor
import os

# Initialize with semantic aggregation
processor = ClaimProcessor(
    use_semantic_aggregation=True,  # Enable semantic mode
    enable_llm_explanations=True,   # Enable LLM explanations
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Process a claim
claim_data = {
    "claim_id": "CLM001",
    "product": "motor",
    "subtype": "accident",
    "claim_amount": 450000,
    "policy_premium": 15000,
    # ... other fields
}

result = processor.process_claim(
    claim_data,
    generate_explanation=True,      # Generate LLM explanation
    explanation_audience="adjuster"  # "adjuster" or "customer"
)

# Access results
print(f"Verdict: {result['verdict']}")  # APPROVE/REVIEW/REJECT
print(f"Confidence: {result['confidence']:.0%}")  # 85%
print(f"Reason: {result['primary_reason']}")
print(f"\nExplanation:\n{result['explanation']}")
```

### Component Verdicts

```python
# Access individual component results
for comp_name, comp_result in result['component_results'].items():
    print(f"{comp_name}:")
    print(f"  Verdict: {comp_result['verdict']}")        # FORGED, HIGH_RISK, etc.
    print(f"  Confidence: {comp_result['confidence']}")
    print(f"  Reason: {comp_result['reason']}")
    print(f"  Red Flags: {comp_result['red_flags']}")
```

### Critical Flags

```python
# Check for critical flags that overrode scoring
if result['critical_flags']:
    for flag in result['critical_flags']:
        print(f"⚠️ {flag['type']}: {flag['reason']}")
        print(f"   Action: {flag['action']}")  # REJECT, REVIEW, FLAG
        print(f"   Override: {flag['override']}")  # True if hard stop
```

### Reasoning Chain

```python
# View full decision-making process
for step in result['reasoning_chain']:
    print(f"Stage: {step['stage']}")
    print(f"Decision: {step['decision']}")
    print(f"Reason: {step['reason']}")
    print(f"Data: {step['data']}")
    print()
```

### Customer-Friendly Explanation

```python
# Generate explanation for customer
result = processor.process_claim(
    claim_data,
    generate_explanation=True,
    explanation_audience="customer"  # Non-technical, empathetic
)

print(result['explanation'])
# Output:
# "Dear Valued Customer,
#  We're unable to approve your claim at this time.
#  Our verification process identified concerns with the submitted documents..."
```

---

## 🔧 Advanced Configuration

### Disable LLM (Use Template Fallback)

```python
# If Groq API is unavailable, system falls back to templates
processor = ClaimProcessor(
    use_semantic_aggregation=True,
    enable_llm_explanations=False  # Use template-based explanations
)
```

### Legacy Mode (Backward Compatible)

```python
# Use old numeric-only aggregation
processor = ClaimProcessor(
    use_semantic_aggregation=False  # Legacy mode
)

result = processor.process_claim(claim_data)
# Returns: {"final_score": 0.73, "verdict": "REJECT", "risk_level": "HIGH"}
```

### Custom LLM Temperature

```python
from src.explainability import FraudExplainer

# Initialize with custom settings
explainer = FraudExplainer(
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,  # More deterministic (default: 0.3)
    max_tokens=300    # Shorter explanations (default: 500)
)

processor = ClaimProcessor(
    use_semantic_aggregation=True,
    enable_llm_explanations=True
)
processor.explainer = explainer  # Override default explainer
```

---

## 📊 Comparison: Legacy vs Semantic

| Feature | Legacy Mode | Semantic Mode |
|---------|-------------|---------------|
| **Output** | Numeric scores only | Semantic verdicts + scores |
| **Explainability** | None | Full reasoning chain |
| **Critical Rules** | No | Yes (forgery, fraud rings) |
| **Weighting** | Fixed | Adaptive (confidence-based) |
| **LLM Explanations** | No | Yes (adjuster + customer) |
| **Backward Compatible** | N/A | Yes |
| **Audit Trail** | Limited | Complete decision steps |

---

## 🐛 Troubleshooting

### Issue: LLM explanations not working

**Solution:**
```bash
# Check API key is set
echo $GROQ_API_KEY

# Verify LangChain is installed
pip show langchain langchain-groq

# Test API key manually
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

### Issue: Rate limit errors

**Solution:**
- Groq free tier: 30 requests/min
- ClaimLens auto-rate-limits to 1 request per 2 seconds
- If still hitting limits, enable caching:

```python
explainer = FraudExplainer(
    enable_caching=True  # Cache similar explanations
)
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip uninstall langchain langchain-groq langchain-core
pip install langchain langchain-groq langchain-core

# Or use requirements
pip install -r requirements.txt --upgrade
```

---

## 🎨 Customization

### Modify Critical Flags

Edit `src/app/semantic_aggregator.py`:

```python
def _check_critical_flags(self, component_results):
    flags = []
    
    # Add your custom rule
    if custom_condition:
        flags.append(CriticalFlag(
            flag_type="YOUR_CUSTOM_FLAG",
            action="REVIEW",
            reason="Your reason here",
            confidence=0.9,
            override=True,  # True = hard stop
            source_component="your_component"
        ))
    
    return flags
```

### Customize LLM Prompts

Edit `src/explainability/llm_explainer.py`:

```python
def _build_adjuster_prompt(self, decision):
    # Modify the prompt template
    template = f"""
    Your custom prompt here...
    Use {decision['verdict']} and {decision['confidence']}
    """
    return ChatPromptTemplate.from_messages([...])
```

### Adjust Weighting

Edit `src/app/semantic_aggregator.py`:

```python
if product_type == "motor":
    base_weights = {
        "document_verification": 0.30,  # Increase from 0.20
        "damage_detection": 0.20,       # Decrease from 0.25
        "ml_fraud_score": 0.35,
        "graph_analysis": 0.15
    }
```

---

## 📝 API Response Format

### Semantic Mode Response

```json
{
  "claim_id": "CLM001",
  "verdict": "REJECT",
  "confidence": 0.92,
  "final_score": 0.78,
  "primary_reason": "Document forgery detected with 92% confidence",
  
  "component_results": {
    "document_verification": {
      "component": "document_verification",
      "verdict": "FORGED",
      "confidence": 0.92,
      "score": 0.92,
      "reason": "PAN card forgery detected",
      "red_flags": ["Document forgery detected (92% confidence)"]
    },
    "ml_fraud_score": {
      "verdict": "HIGH_RISK",
      "confidence": 0.84,
      "score": 0.84,
      "reason": "ML fraud probability 84%"
    }
  },
  
  "critical_flags": [
    {
      "type": "HIGH_CONFIDENCE_FORGERY",
      "action": "REJECT",
      "reason": "Document forgery detected with 92% confidence",
      "confidence": 0.92,
      "override": true
    }
  ],
  
  "reasoning_chain": [
    {
      "stage": "critical_flag_check",
      "decision": "GATING_TRIGGERED",
      "reason": "Found 1 critical flag(s)"
    },
    {
      "stage": "critical_override",
      "decision": "REJECT",
      "reason": "Critical flag overrides scoring: Document forgery..."
    }
  ],
  
  "explanation": "**Fraud Detection Analysis Result**\n\nVerdict: REJECT (Confidence: 92%, Risk Score: 0.95)\n\nPrimary Finding: Document forgery detected with 92% confidence\n\nCritical Issues:\n• Document forgery detected with 92% confidence\n\nRecommended Action: Claim should be rejected pending investigation...",
  
  "fallbacks_used": ["damage_detection"],
  "processing_notes": "Decision overridden by critical flag: HIGH_CONFIDENCE_FORGERY"
}
```

---

## 🚀 Next Steps

1. **Test with Real Data**: Run on your actual claim database
2. **Fine-tune Thresholds**: Adjust critical flag thresholds based on results
3. **Customize Prompts**: Tailor LLM prompts to your business language
4. **Monitor Performance**: Track false positive/negative rates
5. **Integrate into Production**: Add to your claims processing pipeline

---

## 📚 Additional Resources

- [Groq Documentation](https://console.groq.com/docs)
- [LangChain Groq Integration](https://python.langchain.com/docs/integrations/chat/groq)
- [Llama 3.3 Model Card](https://huggingface.co/meta-llama/Llama-3.3-70B)
- [ClaimLens Main README](../README.md)

---

## 💡 Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Run the demo script with debug logging
3. Review the reasoning chain in the output
4. Open an issue on GitHub

---

**Happy Fraud Detecting! 🔍🤖**
