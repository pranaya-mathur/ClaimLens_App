"""
ClaimLens AI v3.0 - UNIFIED FRAUD DETECTION
✅ Uses NEW /api/unified/analyze-complete endpoint
✅ All 4 engines (ML + CV + Graph + LLM) in ONE call
✅ Production-ready with real-time analysis & LLM explanations
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import json
from typing import Dict, Optional
import time

# Configuration - Use 127.0.0.1 instead of localhost
API_URL = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT = f"{API_URL}/api/unified/analyze-complete"

st.set_page_config(
    page_title="ClaimLens AI v3.0 - Unified Fraud Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .verdict-approve {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 8px solid #28a745;
        margin: 1.5rem 0;
    }
    .verdict-review {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 8px solid #ffc107;
        margin: 1.5rem 0;
    }
    .verdict-reject {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 8px solid #dc3545;
        margin: 1.5rem 0;
    }
    .engine-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .flag-critical {
        background: #ffe6e6;
        border-left: 4px solid #ff0000;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .reasoning-chain {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health/", timeout=2)
        return response.status_code == 200
    except Exception as e:
        st.write(f"Debug: {e}")
        return False


def get_verdict_icon(verdict: str) -> str:
    """Get emoji icon for verdict"""
    icons = {
        "APPROVE": "🟢",
        "REVIEW": "🟡",
        "REJECT": "🔴"
    }
    return icons.get(verdict, "⚪")


def get_verdict_color_class(verdict: str) -> str:
    """Get CSS class for verdict"""
    colors = {
        "APPROVE": "verdict-approve",
        "REVIEW": "verdict-review",
        "REJECT": "verdict-reject"
    }
    return colors.get(verdict, "verdict-review")


def create_engine_gauge(score: float, title: str) -> go.Figure:
    """Create gauge chart for component scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def call_unified_api(claim_data: Dict) -> Optional[Dict]:
    """Call unified endpoint with error handling"""
    try:
        with st.spinner("🤖 Running complete fraud analysis with all 4 engines..."):
            # Prepare ISO date format
            claim_data['incident_date'] = claim_data.get('incident_date', str(date.today()))
            
            response = requests.post(
                UNIFIED_ENDPOINT,
                json=claim_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return None
    except requests.Timeout:
        st.error("⏱️ Request timeout - API took too long to respond")
        return None
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        return None


def display_ml_engine_results(ml_result: Dict):
    """Display ML engine results with gauge"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🤖 ML Fraud Scoring")
        ml_score = ml_result.get("confidence", 0)
        
        # Display key metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Fraud Probability", f"{ml_score*100:.1f}%")
        with metric_col2:
            st.metric("Verdict", ml_result.get("verdict", "UNKNOWN"))
        with metric_col3:
            st.metric("Confidence", f"{ml_result.get('confidence', 0)*100:.0f}%")
        
        # Red flags
        if ml_result.get("red_flags"):
            st.markdown("**🚩 Red Flags Detected:**")
            for flag in ml_result.get("red_flags", []):
                st.write(f"• {flag}")
    
    with col2:
        fig = create_engine_gauge(ml_score, "Fraud Probability")
        st.plotly_chart(fig, use_container_width=True)


def display_graph_engine_results(graph_result: Optional[Dict]):
    """Display graph engine results"""
    if not graph_result or graph_result.get("verdict") == "UNAVAILABLE":
        st.info("📊 Graph analysis not available - Neo4j not connected")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🕸️ Graph Network Analysis")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Verdict", graph_result.get("verdict", "UNKNOWN"))
        with metric_col2:
            st.metric("Network Score", f"{graph_result.get('score', 0):.2f}")
        with metric_col3:
            st.metric("Confidence", f"{graph_result.get('confidence', 0)*100:.0f}%")
        
        st.write(f"**Status:** {graph_result.get('reason', 'No information')}")
        
        # Red flags
        if graph_result.get("red_flags"):
            st.markdown("**🚩 Network Red Flags:**")
            for flag in graph_result.get("red_flags", []):
                st.write(f"• {flag}")
    
    with col2:
        fig = create_engine_gauge(graph_result.get("score", 0), "Network Risk")
        st.plotly_chart(fig, use_container_width=True)


def display_reasoning_chain(reasoning_chain: list):
    """Display decision reasoning chain"""
    st.markdown("### 🔗 Decision Reasoning Chain")
    
    for i, step in enumerate(reasoning_chain, 1):
        with st.expander(f"Step {i}: {step.get('stage', 'Unknown').replace('_', ' ').title()}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Decision:** {step.get('decision', 'N/A')}")
            with col2:
                st.write(f"**Confidence:** {step.get('confidence', 0)*100:.0f}%")
            with col3:
                st.write(f"**Reason:** {step.get('reason', 'N/A')}")


def display_llm_explanation(explanation: str, reasoning_chain: list):
    """Display LLM-generated explanation"""
    st.markdown("### 🧠 AI-Generated Explanation (Groq Llama-3.3-70B)")
    
    with st.container():
        st.markdown(f"""
        <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea;">
        {explanation}
        </div>
        """, unsafe_allow_html=True)
    
    # Show reasoning chain
    display_reasoning_chain(reasoning_chain)


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 🤖 ClaimLens v3.0 Status")
    
    # API Health
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ Cannot connect to API")
        st.code(f"# API should be at: {API_URL}\n# Start with:\npython -m uvicorn api.main:app --reload", language="bash")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Components")
    
    try:
        if api_healthy:
            health = requests.get(f"{API_URL}/api/unified/health", timeout=3).json()
            
            st.markdown("**Active Modules:**")
            modules = health.get("modules", {})
            for module, status in modules.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {module.replace('_', ' ').title()}")
    except:
        st.warning("Cannot fetch module status")
    
    st.markdown("---")
    st.markdown("### 📋 v3.0 Features")
    st.markdown("""
    - ✅ **Unified Endpoint** - Single API call
    - ✅ **ML Engine** - CatBoost fraud scoring
    - ✅ **Graph Analysis** - Neo4j fraud rings
    - ✅ **LLM Explanations** - Groq Llama-3.3-70B
    - ✅ **Real-time Results** - All 4 engines
    - ✅ **Auto Storage** - Claims in database
    - ✅ **Reasoning Chain** - Decision transparency
    """)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        [
            "🎯 Claim Analysis",
            "📊 Test Multiple Claims",
            "📈 Analytics"
        ]
    )


# ============================================
# PAGE 1: SINGLE CLAIM ANALYSIS
# ============================================

if "Claim Analysis" in page:
    st.markdown('<p class="main-header">🤖 ClaimLens AI v3.0</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ UNIFIED Fraud Detection | ML + CV + Graph + LLM</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not check_api_health():
        st.error(f"❌ API not running! Make sure API is at {API_URL}")
        st.code("python -m uvicorn api.main:app --reload", language="bash")
    else:
        st.markdown("## 🎯 Analyze Claim (All 4 Engines)")
        
        # Input Form
        with st.expander("📋 Claim Details", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                claim_id = st.text_input("🔖 Claim ID", value=f"CLM{int(time.time())%10000}")
            
            with col2:
                claimant_id = st.text_input("👤 Claimant ID", value="CLMT12345")
            
            with col3:
                policy_id = st.text_input("📜 Policy ID", value="POL98765")
            
            with col4:
                product = st.selectbox("🏢 Product", ["motor", "health", "life", "property"])
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                city = st.text_input("🏙️ City", value="Mumbai")
            
            with col6:
                subtype = st.selectbox(
                    "📦 Claim Type",
                    ["accident", "theft", "fire", "natural_disaster", "medical", "hospitalization"]
                )
            
            with col7:
                claim_amount = st.number_input(
                    "💵 Claim Amount (₹)",
                    min_value=1000,
                    value=250000,
                    step=10000
                )
            
            with col8:
                days_since = st.number_input(
                    "📅 Days Since Policy",
                    min_value=0,
                    value=45,
                    step=1
                )
            
            # Narrative
            narrative = st.text_area(
                "📝 Claim Narrative (Hinglish supported)",
                value="Meri gaadi ko highway pe accident ho gaya. Front bumper and headlight damage hai.",
                height=80
            )
            
            # Documents
            docs = st.text_input(
                "📄 Documents Submitted",
                value="pan,aadhaar,rc,dl,photos",
                help="Comma-separated document types"
            )
        
        st.markdown("---")
        
        # ANALYZE BUTTON
        if st.button("🚀 RUN UNIFIED ANALYSIS", type="primary", use_container_width=True):
            # Prepare request
            claim_data = {
                "claim_id": claim_id,
                "claimant_id": claimant_id,
                "policy_id": policy_id,
                "product": product,
                "city": city,
                "subtype": subtype,
                "claim_amount": float(claim_amount),
                "days_since_policy_start": int(days_since),
                "narrative": narrative,
                "documents_submitted": docs,
                "incident_date": str(date.today())
            }
            
            # Call unified API
            result = call_unified_api(claim_data)
            
            if result:
                st.session_state.last_result = result
                
                # ============================================
                # DISPLAY FINAL VERDICT
                # ============================================
                st.markdown("---")
                st.markdown("## 🎯 FINAL VERDICT")
                
                verdict = result.get("final_verdict", "REVIEW")
                confidence = result.get("final_confidence", 0)
                fraud_prob = result.get("fraud_probability", 0)
                
                verdict_class = get_verdict_color_class(verdict)
                icon = get_verdict_icon(verdict)
                
                st.markdown(f"""
                <div class="{verdict_class}">
                <h2>{icon} {verdict}</h2>
                <p><strong>Claim ID:</strong> {result.get('claim_id')}</p>
                <p><strong>Fraud Probability:</strong> {fraud_prob*100:.1f}%</p>
                <p><strong>Final Confidence:</strong> {confidence*100:.0f}%</p>
                <p><strong>Risk Level:</strong> {result.get('risk_level', 'UNKNOWN')}</p>
                <p><strong>Stored in Database:</strong> {'✅ YES' if result.get('stored_in_database') else '❌ NO'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # ============================================
                # CRITICAL FLAGS
                # ============================================
                critical_flags = result.get("critical_flags", [])
                if critical_flags:
                    st.markdown("## 🚩 CRITICAL FLAGS")
                    for flag in critical_flags:
                        st.markdown(f'<div class="flag-critical"><strong>{flag}</strong></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ============================================
                # ENGINE RESULTS IN TABS
                # ============================================
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🤖 ML Engine",
                    "🕸️ Graph Engine",
                    "🧠 LLM Aggregation",
                    "🔗 Reasoning Chain",
                    "📊 Details"
                ])
                
                with tab1:
                    st.markdown("### 🤖 ML Fraud Scoring")
                    ml_result = result.get("ml_engine", {})
                    display_ml_engine_results(ml_result)
                
                with tab2:
                    st.markdown("### 🕸️ Graph Network Analysis")
                    graph_result = result.get("graph_engine")
                    display_graph_engine_results(graph_result)
                
                with tab3:
                    st.markdown("### 🧠 LLM Semantic Aggregation")
                    llm_agg = result.get("llm_aggregation", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Verdict", llm_agg.get("verdict", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{llm_agg.get('confidence', 0)*100:.0f}%")
                    with col3:
                        st.metric("LLM Used", "✅ YES" if llm_agg.get("llm_used") else "❌ NO")
                
                with tab4:
                    display_reasoning_chain(result.get("reasoning_chain", []))
                
                with tab5:
                    st.markdown("### 📊 Full Response JSON")
                    st.json(result)
                
                st.markdown("---")
                
                # ============================================
                # LLM EXPLANATION
                # ============================================
                explanation = result.get("explanation", "")
                if explanation:
                    display_llm_explanation(explanation, result.get("reasoning_chain", []))


# ============================================
# PAGE 2: BATCH TEST
# ============================================

elif "Test Multiple" in page:
    st.title("📊 Test Multiple Claims")
    st.markdown("Test with pre-configured sample claims to verify all 4 engines working")
    
    if not check_api_health():
        st.error("❌ API not running!")
    else:
        # Sample claims
        sample_claims = {
            "🟢 LOW RISK": {
                "claim_id": "CLM-LOW-001",
                "claimant_id": "CLMT-LOW",
                "policy_id": "POL-LOW",
                "product": "motor",
                "city": "Delhi",
                "subtype": "accident",
                "claim_amount": 50000,
                "days_since_policy_start": 365,
                "narrative": "Small accident, minimal damage, straightforward claim",
                "documents_submitted": "pan,aadhaar,rc"
            },
            "🟡 MEDIUM RISK": {
                "claim_id": "CLM-MED-001",
                "claimant_id": "CLMT-MED",
                "policy_id": "POL-MED",
                "product": "health",
                "city": "Mumbai",
                "subtype": "medical",
                "claim_amount": 500000,
                "days_since_policy_start": 30,
                "narrative": "Hospitalization claim filed early in policy term",
                "documents_submitted": "pan,discharge,bills"
            },
            "🔴 HIGH RISK": {
                "claim_id": "CLM-HIGH-001",
                "claimant_id": "CLMT-HIGH",
                "policy_id": "POL-HIGH",
                "product": "motor",
                "city": "Bangalore",
                "subtype": "theft",
                "claim_amount": 2000000,
                "days_since_policy_start": 10,
                "narrative": "Complete vehicle theft just 10 days after policy activation",
                "documents_submitted": "pan,aadhaar"
            }
        }
        
        # Select claim to test
        selected = st.selectbox("Select Test Claim", list(sample_claims.keys()))
        claim_data = sample_claims[selected]
        claim_data["incident_date"] = str(date.today())
        
        # Show claim details
        with st.expander("📋 Claim Details"):
            st.json(claim_data)
        
        # Test button
        if st.button(f"🚀 Test {selected} Claim", type="primary", use_container_width=True):
            result = call_unified_api(claim_data)
            
            if result:
                # Quick summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Verdict", result.get("final_verdict"))
                with col2:
                    st.metric("Fraud %", f"{result.get('fraud_probability', 0)*100:.0f}%")
                with col3:
                    st.metric("Confidence", f"{result.get('final_confidence', 0)*100:.0f}%")
                with col4:
                    st.metric("Stored", "✅" if result.get("stored_in_database") else "❌")
                
                # Full results
                st.markdown("---")
                st.markdown("### Full Analysis Results")
                st.json(result)


# ============================================
# PAGE 3: ANALYTICS
# ============================================

else:
    st.title("📈 Analytics Dashboard")
    
    if check_api_health():
        try:
            analytics = requests.get(f"{API_URL}/api/analytics/overview", timeout=5).json()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims", f"{analytics.get('total_claims', 0):,}")
            col2.metric("Fraud Rate", f"{analytics.get('fraud_rate', 0):.1f}%")
            col3.metric("Avg Risk Score", f"{analytics.get('avg_fraud_score', 0):.2f}")
            col4.metric("Total Exposed", f"₹{analytics.get('total_exposure', 0)/1000000:.1f}M")
            
        except:
            st.info("Analytics data not available")
    else:
        st.error("API not running")


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <strong>ClaimLens AI v3.0</strong> | Unified Fraud Detection<br>
    ML + CV + Graph + LLM | Built with ❤️ | Production Ready
</div>
""", unsafe_allow_html=True)
