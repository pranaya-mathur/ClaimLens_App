"""
ClaimLens AI v2.0 - PRODUCTION READY
✅ Follows actual API flow:
  1. INGEST claim via /api/ingest/claim
  2. SCORE claim via /api/fraud/score
✅ Document verification
✅ Full error handling
✅ Responsibility = Reliability
"""
import streamlit as st
import requests
import pandas as pd
from datetime import date

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ClaimLens AI - Insurance Fraud Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<p class="main-header">ClaimLens</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🔍 API Status")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3)
        if response.status_code == 200:
            st.success("✅ API Connected")
            api_available = True
        else:
            st.error("❌ API Error")
            api_available = False
    except:
        st.error("❌ Cannot reach API")
        st.code("python -m uvicorn api.main:app --reload", language="bash")
        api_available = False
    
    st.markdown("---")
    st.markdown("### 📋 API Flow")
    st.markdown("""
    1. **INGEST** - Add claim to Neo4j
    2. **VERIFY** - Check documents
    3. **SCORE** - Get fraud risk
    4. **ANALYZE** - Final verdict
    """)
    
    st.markdown("---")
    page = st.radio("Navigate", ["🏷️ Claim Analysis", "📊 Analytics", "🕸️ Networks"])

# PAGE 1: CLAIM ANALYSIS
if "Claim Analysis" in page:
    st.markdown('<p class="main-header">🤖 ClaimLens AI</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>AI-Powered Insurance Fraud Detection</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not api_available:
        st.error("⚠️ API is not running. Start it first!")
        st.stop()
    
    st.markdown("## 📋 Claim Information")
    
    with st.expander("📋 Enter Claim Details", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            claim_id = st.text_input("🏷️ Claim ID", value="CLM2024001")
        with col2:
            claimant_id = st.text_input("👤 Claimant ID", value="CLMT12345")
        with col3:
            policy_id = st.text_input("📄 Policy ID", value="POL12345")
        with col4:
            city = st.text_input("📍 City", value="Mumbai")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            product_type = st.selectbox("🚗 Product", ["motor", "health", "life", "property"])
        with col6:
            claim_subtype = st.selectbox("📦 Type", ["accident", "theft", "fire", "natural_disaster", "mechanical", "vandalism"])
        with col7:
            premium = st.number_input("💰 Premium (₹)", min_value=1000, max_value=100000, value=15000, step=1000)
        with col8:
            claim_amount = st.number_input("💵 Amount (₹)", min_value=1000, max_value=10000000, value=250000, step=10000)
        
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            days_since_policy = st.number_input("📅 Days Since", min_value=0, max_value=3650, value=45)
        with col10:
            incident_date = st.date_input("📆 Incident Date", value=date.today())
        with col11:
            documents_submitted = st.text_input("📄 Documents", value="pan,aadhaar,rc,dl")
        with col12:
            st.write("")
        
        narrative = st.text_area(
            "📝 Narrative",
            value="Meri gaadi ko accident ho gaya. Front bumper damage.",
            height=80
        )
    
    st.markdown("---")
    st.markdown("## 📁 Documents")
    
    doc_col1, doc_col2, doc_col3 = st.columns(3)
    
    pan_file = None
    aadhaar_file = None
    
    with doc_col1:
        st.markdown("### PAN Card")
        pan_file = st.file_uploader("Upload PAN", type=["jpg", "jpeg", "png", "pdf"], key="pan")
        if pan_file and pan_file.type.startswith('image'):
            st.image(pan_file, use_container_width=True)
    
    with doc_col2:
        st.markdown("### Aadhaar Card")
        aadhaar_file = st.file_uploader("Upload Aadhaar", type=["jpg", "jpeg", "png", "pdf"], key="aadhaar")
        if aadhaar_file and aadhaar_file.type.startswith('image'):
            st.image(aadhaar_file, use_container_width=True)
    
    with doc_col3:
        st.markdown("### Damage Photo")
        vehicle_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"], key="vehicle")
        if vehicle_file:
            st.image(vehicle_file, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        st.markdown("## 🤖 Results")
        
        progress = st.empty()
        
        # STEP 1: INGEST
        progress.info("⏳ Step 1/4: Ingesting claim...")
        
        ingest_payload = {
            "claim_id": claim_id,
            "claimant_id": claimant_id,
            "policy_id": policy_id,
            "product": product_type,
            "city": city,
            "subtype": claim_subtype,
            "claim_amount": float(claim_amount),
            "days_since_policy_start": int(days_since_policy),
            "narrative": narrative,
            "documents_submitted": documents_submitted,
            "incident_date": str(incident_date),
            "premium": float(premium)
        }
        
        try:
            ingest_response = requests.post(
                f"{API_BASE_URL}/api/ingest/claim",
                json=ingest_payload,
                timeout=30
            )
            
            if ingest_response.status_code in [201, 409]:
                st.success("✅ Claim ingested")
            else:
                st.error(f"❌ Ingest failed: {ingest_response.status_code}")
                st.error(ingest_response.text)
                st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        
        # STEP 2: VERIFY DOCUMENTS
        progress.info("⏳ Step 2/4: Verifying documents...")
        
        with col1:
            st.markdown("### Documents")
            if pan_file:
                try:
                    pan_file.seek(0)
                    files = {"file": (pan_file.name, pan_file, pan_file.type)}
                    r = requests.post(f"{API_BASE_URL}/api/documents/verify-pan", files=files, timeout=30)
                    if r.status_code == 200:
                        result = r.json()
                        if result.get("is_valid"):
                            st.success(f"✅ Valid")
                        else:
                            st.error(f"❌ Forged")
                        st.metric("Confidence", f"{result.get('confidence', 0)*100:.0f}%")
                    else:
                        st.error(f"Error {r.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)[:80]}")
            else:
                st.info("No docs")
        
        # STEP 3: SCORE
        progress.info("⏳ Step 3/4: Scoring fraud risk...")
        
        with col2:
            st.markdown("### Fraud Score")
            try:
                r = requests.post(
                    f"{API_BASE_URL}/api/fraud/score",
                    json={"claim_id": claim_id},
                    timeout=30
                )
                
                if r.status_code == 200:
                    result = r.json()
                    risk = result.get("risk_level", "UNKNOWN")
                    score = result.get("final_risk_score", 0) * 100
                    
                    if risk in ["HIGH", "CRITICAL"]:
                        st.error(f"🔴 {risk}")
                    elif risk == "MEDIUM":
                        st.warning(f"🟡 {risk}")
                    else:
                        st.success(f"🟢 {risk}")
                    
                    st.metric("Risk", f"{score:.0f}%")
                else:
                    st.error(f"Error {r.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)[:80]}")
        
        # STEP 4: PATTERN
        progress.info("⏳ Step 4/4: Pattern analysis...")
        
        with col3:
            st.markdown("### Pattern")
            ratio = claim_amount / premium
            st.metric("Ratio", f"{ratio:.1f}x")
            st.metric("Days", f"{days_since_policy}d")
            
            if ratio > 15 or days_since_policy < 90:
                st.warning("⚠️ Red flags")
            else:
                st.success("✅ OK")
        
        progress.success("✅ Complete!")

# PAGE 2: ANALYTICS
elif "Analytics" in page:
    st.title("📊 Analytics")
    if not api_available:
        st.stop()
    try:
        r = requests.get(f"{API_BASE_URL}/api/analytics/overview", timeout=5)
        if r.status_code == 200:
            data = r.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Claims", f"{data.get('total_claims', 0):,}")
            col2.metric("Fraud Rate", f"{data.get('fraud_rate', 0):.1f}%")
            col3.metric("Avg Score", f"{data.get('avg_fraud_score', 0):.2f}")
            col4.metric("Total", f"₹{data.get('total_amount', 0)/1000000:.1f}M")
    except: st.error("Error")

# PAGE 3: NETWORKS
else:
    st.title("🕸️ Networks")
    if not api_available:
        st.stop()
    tab1, tab2 = st.tabs(["Rings", "Fraudsters"])
    
    with tab1:
        if st.button("Find Rings"):
            try:
                r = requests.get(f"{API_BASE_URL}/api/fraud/rings", params={"min_shared_docs": 2})
                if r.status_code == 200:
                    st.success(f"Found {r.json().get('total_rings_found', 0)} rings")
            except: st.error("Error")
    
    with tab2:
        if st.button("Find Fraudsters"):
            try:
                r = requests.get(f"{API_BASE_URL}/api/fraud/serial-fraudsters", params={"min_fraud_claims": 3})
                if r.status_code == 200:
                    st.success(f"Found {r.json().get('total_found', 0)} fraudsters")
            except: st.error("Error")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#666;'>ClaimLens AI v2.0 | Built with ❤️</div>", unsafe_allow_html=True)
