#!/usr/bin/env python3
"""
Semantic Fraud Detection Demo

Demonstrates ClaimLens v2.0 features:
- Semantic verdicts instead of just numeric scores
- Critical flag gating logic
- Adaptive weighting based on confidence
- LLM-powered explanations (Groq + Llama-3.3-70B)
- Full reasoning chain for transparency
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app.claim_processor import ClaimProcessor
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def demo_motor_claim_with_forgery():
    """
    Demo 1: Motor claim with suspected document forgery.
    
    This demonstrates:
    - Critical flag for high-confidence forgery
    - Automatic override of normal scoring
    - LLM explanation for adjuster
    """
    print_section("DEMO 1: Motor Claim with Document Forgery")
    
    # Initialize processor with semantic mode
    processor = ClaimProcessor(
        use_semantic_aggregation=True,
        enable_llm_explanations=True,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Sample claim with suspicious PAN card
    claim_data = {
        "claim_id": "CLM2024001",
        "product": "motor",
        "subtype": "accident",
        "claim_amount": 450000,
        "policy_premium": 15000,
        "days_since_policy_start": 45,
        "claimant_id": "CLMT12345",
        "documents_submitted": "pan,aadhaar,rc,dl",
        "damage_photos": []  # Simulating missing photos
    }
    
    print(f"Claim ID: {claim_data['claim_id']}")
    print(f"Product: {claim_data['product'].upper()} / {claim_data['subtype']}")
    print(f"Amount: ₹{claim_data['claim_amount']:,}")
    print(f"Policy Premium: ₹{claim_data['policy_premium']:,}")
    print(f"Claim-to-Premium Ratio: {claim_data['claim_amount']/claim_data['policy_premium']:.1f}x")
    
    # Process claim
    result = processor.process_claim(
        claim_data,
        generate_explanation=True,
        explanation_audience="adjuster"
    )
    
    # Display results
    print("\n--- VERDICT ---")
    print(f"Final Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Risk Score: {result['final_score']:.2f}")
    print(f"\nPrimary Reason: {result['primary_reason']}")
    
    # Component results
    print("\n--- COMPONENT ANALYSIS ---")
    for comp_name, comp_result in result.get('component_results', {}).items():
        print(f"\n{comp_name.replace('_', ' ').title()}:")
        print(f"  Verdict: {comp_result['verdict']}")
        print(f"  Confidence: {comp_result['confidence']:.0%}")
        print(f"  Reason: {comp_result['reason']}")
        if comp_result['red_flags']:
            print(f"  Red Flags: {', '.join(comp_result['red_flags'][:2])}")
    
    # Critical flags
    if result.get('critical_flags'):
        print("\n--- CRITICAL FLAGS ---")
        for flag in result['critical_flags']:
            print(f"  ⚠️  {flag['type']}: {flag['reason']}")
    
    # Reasoning chain
    print("\n--- DECISION REASONING ---")
    for i, step in enumerate(result.get('reasoning_chain', [])[:5], 1):
        print(f"{i}. {step['stage']}: {step['reason']}")
    
    # LLM Explanation
    if result.get('explanation'):
        print("\n--- LLM EXPLANATION (Adjuster) ---")
        print(result['explanation'])
    
    return result

def demo_health_claim_with_fraud_ring():
    """
    Demo 2: Health claim linked to fraud ring.
    
    This demonstrates:
    - Graph analysis detecting fraud ring
    - Critical flag for fraud ring membership
    - Customer-friendly explanation
    """
    print_section("DEMO 2: Health Claim with Fraud Ring Detection")
    
    processor = ClaimProcessor(
        use_semantic_aggregation=True,
        enable_llm_explanations=True,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    claim_data = {
        "claim_id": "CLM2024002",
        "product": "health",
        "subtype": "hospitalization",
        "claim_amount": 245000,
        "policy_premium": 8000,
        "days_since_policy_start": 28,
        "claimant_id": "CLMT67890",
        "hospital_name": "City Hospital XYZ",
        "documents_submitted": "aadhaar,hospital_bill,discharge_summary"
    }
    
    print(f"Claim ID: {claim_data['claim_id']}")
    print(f"Product: {claim_data['product'].upper()} / {claim_data['subtype']}")
    print(f"Hospital: {claim_data['hospital_name']}")
    print(f"Amount: ₹{claim_data['claim_amount']:,}")
    print(f"Days Since Policy: {claim_data['days_since_policy_start']}")
    
    # Process claim
    result = processor.process_claim(
        claim_data,
        generate_explanation=True,
        explanation_audience="customer"  # Customer-friendly explanation
    )
    
    # Display results
    print("\n--- VERDICT ---")
    print(f"Final Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"\nPrimary Reason: {result['primary_reason']}")
    
    # Red flags
    if result.get('red_flags'):
        print("\n--- RED FLAGS ---")
        for flag in result['red_flags'][:5]:
            print(f"  • {flag}")
    
    # Customer explanation
    if result.get('explanation'):
        print("\n--- EXPLANATION FOR CUSTOMER ---")
        print(result['explanation'])
    
    return result

def demo_clean_claim():
    """
    Demo 3: Clean claim that passes all checks.
    
    This demonstrates:
    - Low risk scores across components
    - APPROVE verdict with high confidence
    - Minimal red flags
    """
    print_section("DEMO 3: Clean Motor Claim (Should Approve)")
    
    processor = ClaimProcessor(
        use_semantic_aggregation=True,
        enable_llm_explanations=True,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    claim_data = {
        "claim_id": "CLM2024003",
        "product": "motor",
        "subtype": "accident",
        "claim_amount": 35000,
        "policy_premium": 18000,
        "days_since_policy_start": 456,
        "claimant_id": "CLMT11111",
        "documents_submitted": "pan,aadhaar,rc,dl,fir",
        "damage_photos": []  # Minor claim, photos available
    }
    
    print(f"Claim ID: {claim_data['claim_id']}")
    print(f"Product: {claim_data['product'].upper()} / {claim_data['subtype']}")
    print(f"Amount: ₹{claim_data['claim_amount']:,} (reasonable)")
    print(f"Policy Age: {claim_data['days_since_policy_start']} days")
    print(f"Claim-to-Premium Ratio: {claim_data['claim_amount']/claim_data['policy_premium']:.1f}x")
    
    result = processor.process_claim(
        claim_data,
        generate_explanation=True,
        explanation_audience="adjuster"
    )
    
    print("\n--- VERDICT ---")
    print(f"Final Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Risk Score: {result['final_score']:.2f}")
    
    if result.get('explanation'):
        print("\n--- EXPLANATION ---")
        print(result['explanation'])
    
    return result

def compare_legacy_vs_semantic():
    """
    Demo 4: Compare legacy vs semantic aggregation.
    
    Shows the difference in output format and explainability.
    """
    print_section("DEMO 4: Legacy vs Semantic Aggregation Comparison")
    
    claim_data = {
        "claim_id": "CLM2024004",
        "product": "motor",
        "subtype": "theft",
        "claim_amount": 650000,
        "policy_premium": 20000,
        "days_since_policy_start": 90,
        "claimant_id": "CLMT22222",
        "documents_submitted": "pan,aadhaar,rc",  # Missing FIR!
    }
    
    print("Same claim processed with both modes:\n")
    
    # Legacy mode
    print("--- LEGACY MODE ---")
    processor_legacy = ClaimProcessor(use_semantic_aggregation=False)
    result_legacy = processor_legacy.process_claim(claim_data)
    
    print(f"Final Score: {result_legacy['final_score']:.2f}")
    print(f"Verdict: {result_legacy['verdict']}")
    print(f"Risk Level: {result_legacy['risk_level']}")
    print(f"Fallbacks Used: {len(result_legacy['fallbacks_used'])}")
    
    # Semantic mode
    print("\n--- SEMANTIC MODE ---")
    processor_semantic = ClaimProcessor(
        use_semantic_aggregation=True,
        enable_llm_explanations=False  # Skip LLM for comparison
    )
    result_semantic = processor_semantic.process_claim(claim_data)
    
    print(f"Final Score: {result_semantic['final_score']:.2f}")
    print(f"Verdict: {result_semantic['verdict']}")
    print(f"Confidence: {result_semantic['confidence']:.0%}")
    print(f"Primary Reason: {result_semantic['primary_reason']}")
    print(f"\nCritical Flags: {len(result_semantic.get('critical_flags', []))}")
    print(f"Reasoning Steps: {len(result_semantic.get('reasoning_chain', []))}")
    
    print("\n✅ Semantic mode provides:")
    print("  - Human-readable verdicts (not just numbers)")
    print("  - Confidence levels")
    print("  - Primary reason in plain language")
    print("  - Critical flags that triggered decision")
    print("  - Full reasoning chain for audit")

def main():
    """
    Run all demos.
    """
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  ClaimLens v2.0 - Semantic Fraud Detection Demo".center(78) + "#")
    print("#" + "  Powered by Groq + Llama-3.3-70B".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        print("\n⚠️  WARNING: GROQ_API_KEY not set. LLM explanations will use templates.")
        print("   Get free API key from: https://console.groq.com/\n")
    
    try:
        # Run demos
        demo_motor_claim_with_forgery()
        input("\n\nPress Enter to continue to next demo...")
        
        demo_health_claim_with_fraud_ring()
        input("\n\nPress Enter to continue to next demo...")
        
        demo_clean_claim()
        input("\n\nPress Enter to continue to comparison demo...")
        
        compare_legacy_vs_semantic()
        
        print_section("ALL DEMOS COMPLETE")
        print("✅ ClaimLens v2.0 semantic aggregation and LLM explanations working!")
        print("\nNext steps:")
        print("  1. Set GROQ_API_KEY in .env for LLM explanations")
        print("  2. Test with your own claim data")
        print("  3. Integrate into your application")
        print("  4. Customize prompts in llm_explainer.py for your needs\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
