#!/usr/bin/env python3
"""
Script to update Streamlit v2 SOTA to use the new unified endpoint.

Run this after pulling latest code:
    python scripts/update_streamlit_unified.py
"""
import re

STREAMLIT_FILE = "frontend/streamlit_app_v2_sota.py"

print("🔧 Updating Streamlit v2 to use unified endpoint...")

# Read current file
with open(STREAMLIT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# PATCH 1: Change ML API call to unified endpoint
old_ml_call = 'ml_response = requests.post(f"{API_URL}/api/ml/score/detailed", json=ml_payload, timeout=30)'
new_ml_call = '''# 🎯 UNIFIED ENDPOINT - All modules in one call!
            unified_payload = {
                "claim_id": claim_id,
                "claimant_id": claimant_id,
                "policy_id": f"POL{claimant_id[4:]}",
                "product": product,
                "city": "Mumbai",
                "subtype": subtype,
                "claim_amount": float(claim_amount),
                "days_since_policy_start": int(days_since_policy),
                "narrative": narrative,
                "documents_submitted": documents,
                "incident_date": str(date.today())
            }
            
            ml_response = requests.post(f"{API_URL}/api/unified/analyze-complete", json=unified_payload, timeout=60)'''

if old_ml_call in content:
    content = content.replace(old_ml_call, new_ml_call)
    print("✅ Patched: ML API call → Unified endpoint")
else:
    print("⚠️ ML API call pattern not found")

# PATCH 2: Update result handling to use unified response
old_result_handling = '''if ml_response.status_code == 200:
                ml_data = ml_response.json()
                result['component_results']['ml_fraud_score'] = {
                    'verdict': ml_data['risk_level'],
                    'confidence': ml_data['fraud_probability'],
                    'score': ml_data['fraud_probability'],
                    'reason': f"ML fraud probability {ml_data['fraud_probability']:.0%}",
                    'red_flags': [f"Risk level: {ml_data['risk_level']}"]
                }'''

new_result_handling = '''if ml_response.status_code == 200:
                # 🎯 UNIFIED RESPONSE - Contains all module results!
                unified_data = ml_response.json()
                
                # Update result with unified data
                result['verdict'] = unified_data['final_verdict']
                result['confidence'] = unified_data['final_confidence']
                result['final_score'] = unified_data['fraud_probability']
                result['primary_reason'] = unified_data['explanation'][:100]
                result['explanation'] = unified_data['explanation']
                result['reasoning_chain'] = unified_data['reasoning_chain']
                result['critical_flags'] = unified_data['critical_flags']
                
                # Component results
                result['component_results']['ml_fraud_score'] = unified_data['ml_engine']
                
                if unified_data.get('cv_engine'):
                    result['component_results']['cv_verification'] = unified_data['cv_engine']
                
                if unified_data.get('graph_engine'):
                    result['component_results']['graph_analysis'] = unified_data['graph_engine']
                
                if unified_data.get('llm_aggregation'):
                    result['component_results']['llm_aggregation'] = unified_data['llm_aggregation']
                
                # Storage status
                if unified_data.get('stored_in_database'):
                    st.success(f"✅ Claim stored in Neo4j at {unified_data.get('storage_timestamp', 'N/A')}")
                else:
                    st.info("📦 Neo4j not available - claim not persisted")'''

if old_result_handling in content:
    content = content.replace(old_result_handling, new_result_handling)
    print("✅ Patched: Result handling → Unified response format")
else:
    print("⚠️ Result handling pattern not found - manual update needed")

# Write updated file
with open(STREAMLIT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Updated {STREAMLIT_FILE}")
print("")
print("🎯 Now your Streamlit calls:")
print("  POST /api/unified/analyze-complete")
print("")
print("🚀 Restart Streamlit to see changes:")
print("  streamlit run frontend/streamlit_app_v2_sota.py")
