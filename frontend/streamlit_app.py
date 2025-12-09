"""
ClaimLens Streamlit Dashboard
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ClaimLens - Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 ClaimLens - AI Fraud Detection Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    page = st.selectbox(
        "Select Page",
        ["Fraud Score Check", "Fraud Rings", "Serial Fraudsters", "Analytics"]
    )

# Page 1: Fraud Score Check
if page == "Fraud Score Check":
    st.header("🎯 Check Claim Fraud Risk")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        claim_id = st.number_input(
            "Enter Claim ID",
            min_value=8000001,
            max_value=9000000,
            value=8000001,
            step=1
        )
        
        if st.button("🔍 Analyze Fraud Risk", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/fraud/score",
                        json={"claim_id": claim_id}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        with col2:
                            # Risk Score Display
                            risk_score = data["final_risk_score"]
                            risk_level = data["risk_level"]
                            
                            # Color based on risk
                            if risk_level == "CRITICAL":
                                color = "🔴"
                            elif risk_level == "HIGH":
                                color = "🟠"
                            elif risk_level == "MEDIUM":
                                color = "🟡"
                            else:
                                color = "🟢"
                            
                            st.markdown(f"## {color} Risk Level: **{risk_level}**")
                            st.metric("Final Risk Score", f"{risk_score:.2%}")
                            
                            # Score breakdown
                            st.subheader("📊 Score Breakdown")
                            
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Base Fraud Score", f"{data['base_fraud_score']:.2%}")
                            col_b.metric("Claimant Fraud Rate", 
                                       f"{data['graph_insights']['claimant_fraud_rate']:.2%}")
                            col_c.metric("Neighbor Frauds", 
                                       data['graph_insights']['neighbor_fraud_count'])
                            
                            # Recommendation
                            st.markdown("---")
                            st.subheader("💡 Recommendation")
                            st.info(data['recommendation'])
                            
                            # Graph Insights
                            st.markdown("---")
                            st.subheader("🕸️ Graph Insights")
                            insights = data['graph_insights']
                            st.write(f"- **Document Sharing**: {insights['doc_sharing_count']} claims")
                            st.write(f"- **Neighbor Fraud Count**: {insights['neighbor_fraud_count']} fraudulent neighbors")
                    
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# Page 2: Fraud Rings
elif page == "Fraud Rings":
    st.header("🕸️ Fraud Ring Detection")
    st.markdown("Find claimants sharing documents (potential collusion)")
    
    min_docs = st.slider("Minimum Shared Documents", 2, 10, 2)
    
    if st.button("🔍 Find Fraud Rings"):
        with st.spinner("Searching..."):
            try:
                response = requests.get(
                    f"{API_URL}/api/fraud/rings",
                    params={"min_shared_docs": min_docs}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found {data['total_rings_found']} fraud rings!")
                    
                    if data['rings']:
                        df = pd.DataFrame(data['rings'])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No fraud rings found with current criteria")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page 3: Serial Fraudsters
elif page == "Serial Fraudsters":
    st.header("👤 Serial Fraudster Detection")
    st.markdown("Find claimants with multiple high-fraud claims")
    
    min_claims = st.slider("Minimum Fraud Claims", 2, 10, 3)
    
    if st.button("🔍 Find Serial Fraudsters"):
        with st.spinner("Searching..."):
            try:
                response = requests.get(
                    f"{API_URL}/api/fraud/serial-fraudsters",
                    params={"min_fraud_claims": min_claims}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found {data['total_found']} serial fraudsters!")
                    
                    if data['fraudsters']:
                        df = pd.DataFrame(data['fraudsters'])
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualization
                        fig = px.bar(
                            df.head(20),
                            x='claimant_id',
                            y='total_claimed',
                            title="Top 20 Serial Fraudsters by Amount Claimed"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No serial fraudsters found")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page 4: Analytics
elif page == "Analytics":
    st.header("📊 Fraud Analytics Dashboard")
    
    # Mock data - replace with real API calls
    st.subheader("Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims", "50,000")
    col2.metric("Fraud Rate", "11.2%")
    col3.metric("Avg Processing Time", "1.8s")
    col4.metric("Auto-Approved", "62%")
    
    st.markdown("---")
    st.info("📈 Advanced analytics coming soon...")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        ClaimLens v1.0 | Built with ❤️ by Pranaya & Team
    </div>
    """,
    unsafe_allow_html=True
)
