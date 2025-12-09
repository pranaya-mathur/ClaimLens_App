"""
ClaimLens Streamlit Dashboard
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                            x='p.claimant_id',
                            y='total_claimed',
                            title="Top 20 Serial Fraudsters by Amount Claimed"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No serial fraudsters found")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page 4: Advanced Analytics
elif page == "Analytics":
    st.header("📊 Advanced Fraud Analytics")
    
    # Fetch overview stats
    try:
        overview_response = requests.get(f"{API_URL}/api/analytics/overview")
        
        if overview_response.status_code == 200:
            overview = overview_response.json()
            
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Total Claims",
                f"{overview['total_claims']:,}",
                help="Total number of claims in database"
            )
            col2.metric(
                "Fraud Claims",
                f"{overview['fraud_claims']:,}",
                f"{overview['fraud_rate']}%",
                help="Number and percentage of confirmed fraud"
            )
            col3.metric(
                "Avg Fraud Score",
                f"{overview['avg_fraud_score']:.3f}",
                help="Average fraud risk score across all claims"
            )
            col4.metric(
                "Total Amount",
                f"₹{overview['total_amount']/1000000:.1f}M",
                help="Total claim amount processed"
            )
            
            st.markdown("---")
            
            # Row 1: Risk Distribution & Product Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Risk Level Distribution")
                risk_response = requests.get(f"{API_URL}/api/analytics/risk-distribution")
                if risk_response.status_code == 200:
                    risk_data = pd.DataFrame(risk_response.json())
                    
                    # Pie chart
                    fig_pie = px.pie(
                        risk_data,
                        values='count',
                        names='risk_level',
                        color='risk_level',
                        color_discrete_map={
                            'CRITICAL': '#FF4B4B',
                            'HIGH': '#FFA500',
                            'MEDIUM': '#FFD700',
                            'LOW': '#90EE90'
                        },
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("💼 Fraud Rate by Product")
                product_response = requests.get(f"{API_URL}/api/analytics/by-product")
                if product_response.status_code == 200:
                    product_data = pd.DataFrame(product_response.json())
                    
                    # Bar chart
                    fig_product = px.bar(
                        product_data,
                        x='product',
                        y='fraud_rate',
                        color='fraud_rate',
                        color_continuous_scale='Reds',
                        labels={'fraud_rate': 'Fraud Rate (%)', 'product': 'Product Type'},
                        text='fraud_rate'
                    )
                    fig_product.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_product.update_layout(showlegend=False)
                    st.plotly_chart(fig_product, use_container_width=True)
            
            st.markdown("---")
            
            # Row 2: City Analysis
            st.subheader("🏛️ Top 10 High-Risk Cities")
            city_response = requests.get(f"{API_URL}/api/analytics/by-city")
            if city_response.status_code == 200:
                city_data = pd.DataFrame(city_response.json())
                
                # Horizontal bar chart
                fig_city = px.bar(
                    city_data,
                    y='city',
                    x='fraud_rate',
                    orientation='h',
                    color='fraud_rate',
                    color_continuous_scale='OrRd',
                    labels={'fraud_rate': 'Fraud Rate (%)', 'city': 'City'},
                    text='fraud_rate'
                )
                fig_city.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_city.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_city, use_container_width=True)
                
                # Show data table
                with st.expander("📊 View Detailed City Data"):
                    st.dataframe(
                        city_data.style.background_gradient(subset=['fraud_rate'], cmap='Reds'),
                        use_container_width=True
                    )
        
        else:
            st.error("Failed to fetch analytics data. Make sure API is running.")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
        st.info("⚠️ Make sure the API server is running and analytics endpoints are available.")

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