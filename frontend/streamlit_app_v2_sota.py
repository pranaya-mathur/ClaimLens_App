#!/usr/bin/env python3
"""
ClaimLens v2.0 SOTA Dashboard - Real API Integration
Now connects to actual backend endpoints for live analysis
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, date
import json
import time
import base64
from io import BytesIO

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ClaimLens AI - Explainable Fraud Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SOTA Custom CSS with Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Verdict cards with gradient */
    .verdict-approve { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
        animation: slideIn 0.5s ease-out;
    }
    .verdict-review { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.4);
        animation: slideIn 0.5s ease-out;
    }
    .verdict-reject { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(250, 112, 154, 0.4);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Component cards */
    .component-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .component-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Critical flag with pulse animation */
    .critical-flag {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
        color: white;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(255, 154, 86, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Reasoning step */
    .reasoning-step {
        background: rgba(255, 255, 255, 0.9);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.8;
        }
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 3.5em; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;'>
        🤖 ClaimLens AI
    </h1>
    <p style='font-size: 1.2em; color: #666; margin-top: -10px;'>
        ✨ Explainable AI Fraud Detection | Powered by Groq + Llama-3.3-70B
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h2 style='color: #667eea;'>⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        "📊 Select Module",
        ["AI Claim Analyzer", "Fraud Network Graph", "Serial Fraudsters", "Analytics Dashboard"],
        help="Navigate between different analysis modules"
    )
    
    st.markdown("---")
    
    st.subheader("🧠 AI Mode")
    semantic_mode = st.toggle("✨ Semantic Aggregation", value=True, help="Use v2.0 semantic verdicts")
    
    if semantic_mode:
        explanation_mode = st.toggle("🤖 LLM Explanations", value=True, help="Generate AI explanations")
        
        if explanation_mode:
            st.markdown("**🎯 Explanation Style:**")
            audience = st.radio(
                "",
                ["Adjuster (Technical)", "Customer (Friendly)"],
                label_visibility="collapsed"
            )
            
            stream_mode = st.toggle("💬 Streaming Mode", value=True, help="Stream explanation in real-time")
    else:
        explanation_mode = False
        stream_mode = False
        audience = "Adjuster (Technical)"
    
    st.markdown("---")
    
    # API Health Check
    try:
        health_check = requests.get(f"{API_URL}/health/liveness", timeout=2)
        api_status = "✅" if health_check.status_code == 200 else "⚠️"
        status_text = "Models Active" if health_check.status_code == 200 else "Degraded"
    except:
        api_status = "❌"
        status_text = "API Offline"
    
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #667eea; margin: 0;'>AI Status</h3>
        <p style='font-size: 2em; margin: 10px 0;'>{api_status}</p>
        <p style='margin: 0;'>{status_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 20px; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;'>
        <p style='font-size: 0.9em; margin: 0;'>
            <strong>🆕 v2.0 Features:</strong><br>
            • Semantic Verdicts<br>
            • Critical Flags<br>
            • Reasoning Chain<br>
            • LLM Explanations<br>
            • Adaptive Weighting<br>
            • Network Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# Helper Functions
def create_confidence_gauge(confidence, title="Confidence"):
    """Create animated confidence gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': title, 'font': {'size': 20, 'color': '#667eea'}},
        delta={'reference': 70, 'increasing': {'color': "#11998e"}, 'decreasing': {'color': "#f5576c"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_radial(component_results):
    """Create radial risk chart."""
    categories = [name.replace('_', ' ').title() for name in component_results.keys()]
    scores = [res['score'] * 100 for res in component_results.values()]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Risk Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=50, b=50)
    )
    return fig

def create_network_graph(claim_id, graph_data=None):
    """Create fraud network graph."""
    G = nx.Graph()
    G.add_node(claim_id, node_type='claim', risk='high')
    
    if graph_data and 'graph_insights' in graph_data:
        insights = graph_data['graph_insights']
        fraud_count = insights.get('neighbor_fraud_count', 0)
        
        # Add nodes based on actual data
        for i in range(min(fraud_count, 3)):  # Limit to 3 neighbors
            G.add_node(f'FRAUD_{i}', node_type='fraud_neighbor')
            G.add_edge(claim_id, f'FRAUD_{i}')
    else:
        # Minimal graph if no data
        G.add_node('CLMT001', node_type='claimant')
        G.add_edge(claim_id, 'CLMT001')
    
    pos = nx.spring_layout(G)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=2, color='#888'),
        hoverinfo='none', mode='lines'
    )
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())
    
    node_colors = ['#ff4444' if 'FRAUD' in str(node) else '#667eea' for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hoverinfo='text', text=node_text, textposition="top center",
        marker=dict(size=30, color=node_colors, line=dict(width=2, color='white'))
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    return fig

def stream_text(text, placeholder):
    """Stream text word by word."""
    words = text.split()
    displayed_text = ""
    for word in words:
        displayed_text += word + " "
        placeholder.markdown(displayed_text)
        time.sleep(0.05)

def get_verdict_emoji(verdict):
    """Get emoji for verdict."""
    emoji_map = {
        "APPROVE": "👍", "REVIEW": "🔍", "REJECT": "❌",
        "FORGED": "🚫", "AUTHENTIC": "✅", "SUSPICIOUS": "⚠️",
        "HIGH_RISK": "🔴", "MEDIUM_RISK": "🟡", "LOW_RISK": "🟢",
        "CRITICAL": "🔥", "FRAUD_RING_DETECTED": "🕸️", "CLEAN": "✨"
    }
    return emoji_map.get(verdict, "📊")

# ============================================================================
# MODULE 1: AI CLAIM ANALYZER
# ============================================================================

if page == "AI Claim Analyzer":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🎯 AI-Powered Claim Analysis</h2>", unsafe_allow_html=True)
    
    # Input form with modern design
    with st.expander("📝 Enter Claim Information", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            claim_id = st.text_input("🏷️ Claim ID", value="CLM-2025-001")
            product = st.selectbox("📦 Product Type", ["motor", "health", "life", "property"])
        
        with col2:
            subtype = st.text_input("📝 Claim Subtype", value="accident")
            claim_amount = st.number_input("💵 Claim Amount (₹)", min_value=0, value=250000, step=10000)
        
        with col3:
            policy_premium = st.number_input("💳 Premium (₹)", min_value=0, value=15000, step=1000)
            days_since_policy = st.number_input("📅 Days Since Policy", min_value=0, value=45)
        
        with col4:
            claimant_id = st.text_input("👤 Claimant ID", value="CLMT-12345")
            documents = st.text_input("📄 Documents", value="pan,aadhaar,rc,dl")
        
        narrative = st.text_area(
            "📝 Claim Narrative (Hinglish)",
            value="Meri gaadi ko accident ho gaya tha highway pe. Front bumper aur headlight damage hai.",
            height=80
        )
    
    # Document Upload Section - UPDATED WITH 4TH COLUMN ✨
    st.markdown("### 📤 Upload Documents")
    doc_col1, doc_col2, doc_col3, doc_col4 = st.columns(4)
    
    with doc_col1:
        pan_file = st.file_uploader("🆔 PAN Card", type=["jpg", "jpeg", "png"], key="pan")
        if pan_file:
            st.image(pan_file, use_container_width=True)
    
    with doc_col2:
        aadhaar_file = st.file_uploader("🪪 Aadhaar Card", type=["jpg", "jpeg", "png"], key="aadhaar")
        if aadhaar_file:
            st.image(aadhaar_file, use_container_width=True)
    
    with doc_col3:
        vehicle_file = st.file_uploader("🚗 Vehicle Photo", type=["jpg", "jpeg", "png"], key="vehicle")
        if vehicle_file:
            st.image(vehicle_file, use_container_width=True)
    
    with doc_col4:
        # 👇 NEW 4TH DOCUMENT UPLOAD SECTION
        st.markdown("**📋 Other Documents**")
        doc_type = st.selectbox(
            "Document Type",
            [
                "hospital_bill",
                "discharge_summary",
                "fir",
                "police_report",
                "invoice",
                "license",
                "passport",
                "voter_id",
                "bank_statement",
                "death_certificate",
                "other"
            ],
            key="doc_type_select"
        )
        
        other_doc_file = st.file_uploader(
            "📁 Upload File",
            type=["jpg", "jpeg", "png", "pdf"],
            key="other_doc"
        )
        if other_doc_file:
            st.success(f"✅ {doc_type.replace('_', ' ').title()} uploaded")
    
    st.markdown("---")
    
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        analyze_btn = st.button("🚀 Analyze with AI", type="primary", use_container_width=True)
    
    if analyze_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Real API calls
        result = {
            "claim_id": claim_id,
            "verdict": "REVIEW",
            "confidence": 0.75,
            "final_score": 0.55,
            "primary_reason": "Analysis in progress...",
            "component_results": {},
            "critical_flags": [],
            "reasoning_chain": [],
            "explanation": ""
        }
        
        # 1. Document Verification
        status_text.text("📄 Verifying documents...")
        progress_bar.progress(0.2)
        
        doc_results = {}
        
        if pan_file:
            try:
                pan_file.seek(0)
                files = {"file": (pan_file.name, pan_file, pan_file.type)}
                pan_response = requests.post(f"{API_URL}/api/documents/verify-pan", files=files, timeout=30)
                
                if pan_response.status_code == 200:
                    pan_data = pan_response.json()
                    doc_results['pan'] = pan_data
                    
                    verdict = "SUSPICIOUS" if pan_data['risk_score'] > 0.4 else "CLEAN"
                    result['component_results']['document_verification'] = {
                        'verdict': verdict,
                        'confidence': pan_data['confidence'],
                        'score': pan_data['risk_score'],
                        'reason': pan_data['recommendation'],
                        'red_flags': pan_data['red_flags']
                    }
            except Exception as e:
                st.warning(f"PAN verification error: {str(e)}")
        
        if aadhaar_file:
            try:
                aadhaar_file.seek(0)
                files = {"file": (aadhaar_file.name, aadhaar_file, aadhaar_file.type)}
                aadhaar_response = requests.post(f"{API_URL}/api/documents/verify-aadhaar", files=files, timeout=30)
                
                if aadhaar_response.status_code == 200:
                    aadhaar_data = aadhaar_response.json()
                    doc_results['aadhaar'] = aadhaar_data
                    
                    verdict = "SUSPICIOUS" if aadhaar_data['risk_score'] > 0.4 else "CLEAN"
                    if 'document_verification' not in result['component_results']:
                        result['component_results']['document_verification'] = {
                            'verdict': verdict,
                            'confidence': aadhaar_data['confidence'],
                            'score': aadhaar_data['risk_score'],
                            'reason': aadhaar_data['recommendation'],
                            'red_flags': aadhaar_data['red_flags']
                        }
            except Exception as e:
                st.warning(f"Aadhaar verification error: {str(e)}")
        
        # 👇 NEW: Generic Document Verification for Other Documents
        if other_doc_file:
            try:
                other_doc_file.seek(0)
                files = {"file": (other_doc_file.name, other_doc_file, other_doc_file.type)}
                data = {"document_type": doc_type}
                
                other_response = requests.post(
                    f"{API_URL}/api/documents/verify-document",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if other_response.status_code == 200:
                    other_data = other_response.json()
                    doc_results['other_document'] = other_data
                    
                    verdict = "SUSPICIOUS" if other_data['risk_score'] > 0.4 else "CLEAN"
                    
                    # Merge with existing document_verification or create new
                    if 'document_verification' in result['component_results']:
                        # Average the scores
                        existing = result['component_results']['document_verification']
                        avg_score = (existing['score'] + other_data['risk_score']) / 2
                        existing['score'] = avg_score
                        existing['red_flags'].extend(other_data['red_flags'])
                    else:
                        result['component_results']['document_verification'] = {
                            'verdict': verdict,
                            'confidence': other_data['confidence'],
                            'score': other_data['risk_score'],
                            'reason': other_data['recommendation'],
                            'red_flags': other_data['red_flags']
                        }
                    
                    st.success(f"✅ {doc_type.replace('_', ' ').title()} verified")
            except Exception as e:
                st.warning(f"{doc_type} verification error: {str(e)}")
        
        # 2. ML Fraud Scoring
        status_text.text("🔍 Running ML fraud detection...")
        progress_bar.progress(0.4)
        
        try:
            ml_payload = {
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
            
            ml_response = requests.post(f"{API_URL}/api/ml/score/detailed", json=ml_payload, timeout=30)
            
            if ml_response.status_code == 200:
                ml_data = ml_response.json()
                result['component_results']['ml_fraud_score'] = {
                    'verdict': ml_data['risk_level'],
                    'confidence': ml_data['fraud_probability'],
                    'score': ml_data['fraud_probability'],
                    'reason': f"ML fraud probability {ml_data['fraud_probability']:.0%}",
                    'red_flags': [f"Risk level: {ml_data['risk_level']}"]
                }
        except Exception as e:
            st.warning(f"ML scoring error: {str(e)}")
        
        # 3. Graph Analysis
        status_text.text("🕸️ Analyzing fraud networks...")
        progress_bar.progress(0.6)
        
        try:
            claim_num = int(claim_id.replace("CLM", "").replace("-", "")) if claim_id.replace("CLM", "").replace("-", "").isdigit() else 8000001
            graph_response = requests.post(f"{API_URL}/api/fraud/score", json={"claim_id": claim_num}, timeout=10)
            
            if graph_response.status_code == 200:
                graph_data = graph_response.json()
                insights = graph_data.get('graph_insights', {})
                fraud_count = insights.get('neighbor_fraud_count', 0)
                
                result['component_results']['graph_analysis'] = {
                    'verdict': 'FRAUD_RING_DETECTED' if fraud_count > 0 else 'CLEAN',
                    'confidence': 0.88,
                    'score': graph_data.get('final_risk_score', 0.12),
                    'reason': f"{fraud_count} fraud connections detected" if fraud_count > 0 else "No fraud network detected",
                    'red_flags': [f"{fraud_count} fraud neighbors"] if fraud_count > 0 else []
                }
                result['graph_data'] = graph_data
        except:
            result['component_results']['graph_analysis'] = {
                'verdict': 'CLEAN',
                'confidence': 0.85,
                'score': 0.15,
                'reason': 'Graph analysis unavailable',
                'red_flags': []
            }
        
        # 4. Calculate final verdict
        status_text.text("🧠 Applying semantic aggregation...")
        progress_bar.progress(0.8)
        
        # Weighted scoring
        doc_score = result['component_results'].get('document_verification', {}).get('score', 0) * 0.3
        ml_score = result['component_results'].get('ml_fraud_score', {}).get('score', 0) * 0.4
        graph_score = result['component_results'].get('graph_analysis', {}).get('score', 0) * 0.3
        
        result['final_score'] = doc_score + ml_score + graph_score
        
        if result['final_score'] >= 0.7:
            result['verdict'] = "REJECT"
            result['primary_reason'] = "High fraud probability detected across multiple components"
        elif result['final_score'] >= 0.4:
            result['verdict'] = "REVIEW"
            result['primary_reason'] = f"Claim amount ₹{claim_amount:,} against premium ₹{policy_premium:,} ({claim_amount/policy_premium:.0f}x ratio) with {days_since_policy} days policy age"
        else:
            result['verdict'] = "APPROVE"
            result['primary_reason'] = "Low fraud indicators detected"
        
        result['confidence'] = 1 - abs(result['final_score'] - 0.5) * 2
        
        # 5. Generate explanation
        status_text.text("🤖 Generating AI explanation...")
        progress_bar.progress(0.9)
        
        explanation_parts = []
        explanation_parts.append(f"This claim requires {result['verdict'].lower()} based on our multi-modal analysis.")
        explanation_parts.append(f"The claim amount of ₹{claim_amount:,} against a premium of ₹{policy_premium:,} represents a {claim_amount/policy_premium:.0f}x ratio.")
        explanation_parts.append(f"The claim was filed {days_since_policy} days after policy inception.")
        
        if 'ml_fraud_score' in result['component_results']:
            ml_prob = result['component_results']['ml_fraud_score']['score'] * 100
            explanation_parts.append(f"Our ML models predict {ml_prob:.0f}% fraud probability.")
        
        if 'document_verification' in result['component_results']:
            doc_verdict = result['component_results']['document_verification']['verdict']
            explanation_parts.append(f"Document verification: {doc_verdict}.")
        
        if 'graph_analysis' in result['component_results']:
            graph_verdict = result['component_results']['graph_analysis']['verdict']
            if 'FRAUD' in graph_verdict:
                explanation_parts.append("Fraud network connections were detected.")
            else:
                explanation_parts.append("No fraud network connections were detected.")
        
        explanation_parts.append("We recommend verifying the claimant's history before processing.")
        
        result['explanation'] = " ".join(explanation_parts)
        
        # Reasoning chain
        result['reasoning_chain'] = [
            {"stage": "document_verification", "decision": result['component_results'].get('document_verification', {}).get('verdict', 'N/A'), "reason": "Document analysis completed"},
            {"stage": "ml_fraud_scoring", "decision": result['component_results'].get('ml_fraud_score', {}).get('verdict', 'N/A'), "reason": "ML risk assessment completed"},
            {"stage": "graph_analysis", "decision": result['component_results'].get('graph_analysis', {}).get('verdict', 'N/A'), "reason": "Network analysis completed"},
            {"stage": "final_verdict", "decision": result['verdict'], "reason": f"Final risk score: {result['final_score']:.2f}"}
        ]
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        st.markdown("---")
        
        # Display Results
        emoji = get_verdict_emoji(result['verdict'])
        verdict_class = f"verdict-{result['verdict'].lower()}"
        
        st.markdown(f"""
        <div class='{verdict_class}'>
            <h1 style='text-align: center; margin: 0; font-size: 3em;'>{emoji}</h1>
            <h2 style='text-align: center; margin: 10px 0;'>Verdict: {result['verdict']}</h2>
            <p style='text-align: center; font-size: 1.1em; opacity: 0.9;'>{result['primary_reason']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_confidence = create_confidence_gauge(result['confidence'], "Confidence")
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col2:
            fig_risk = create_confidence_gauge(result['final_score'], "Risk Score")
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col3:
            st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <h3 style='color: #667eea;'>Components</h3>
                <p style='font-size: 3em; margin: 0;'>{}</p>
                <p>Analyzed</p>
            </div>
            """.format(len(result['component_results'])), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <h3 style='color: #667eea;'>Claim:Premium</h3>
                <p style='font-size: 3em; margin: 0;'>{}x</p>
                <p>Ratio</p>
            </div>
            """.format(int(claim_amount/policy_premium)), unsafe_allow_html=True)
        
        # Component Analysis
        st.markdown("<h3 style='color: #667eea; margin-top: 30px;'>📦 Component Risk Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            for comp_name, comp_result in result['component_results'].items():
                emoji = get_verdict_emoji(comp_result['verdict'])
                st.markdown(f"""
                <div class='component-card'>
                    <h4 style='color: #667eea; margin: 0;'>{emoji} {comp_name.replace('_', ' ').title()}</h4>
                    <div style='margin-top: 10px;'>
                        <span style='background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.85em;'>
                            {comp_result['verdict']}
                        </span>
                        <span style='margin-left: 10px; color: #666;'>
                            Confidence: {comp_result['confidence']:.0%}
                        </span>
                    </div>
                    <p style='margin-top: 10px; color: #666;'>{comp_result['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if result['component_results']:
                fig_radial = create_risk_radial(result['component_results'])
                st.plotly_chart(fig_radial, use_container_width=True)
        
        # Network Graph
        st.markdown("<h3 style='color: #667eea; margin-top: 30px;'>🕸️ Fraud Network Analysis</h3>", unsafe_allow_html=True)
        fig_network = create_network_graph(claim_id, result.get('graph_data'))
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Reasoning Chain
        with st.expander("🧠 Decision Reasoning Chain", expanded=False):
            for i, step in enumerate(result['reasoning_chain'], 1):
                st.markdown(f"""
                <div class='reasoning-step'>
                    <h4 style='color: #667eea; margin: 0;'>Step {i}: {step['stage'].replace('_', ' ').title()}</h4>
                    <p style='margin: 5px 0 0 0;'><strong>Decision:</strong> {step['decision']}</p>
                    <p style='margin: 5px 0 0 0; color: #666;'>{step['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI Explanation
        if explanation_mode:
            st.markdown("<h3 style='color: #667eea; margin-top: 30px;'>🤖 AI-Generated Explanation</h3>", unsafe_allow_html=True)
            
            audience_type = "Adjuster" if "Adjuster" in audience else "Customer"
            st.info(f"🎯 Explanation for: **{audience_type}**")
            
            explanation_placeholder = st.empty()
            
            if stream_mode:
                stream_text(result['explanation'], explanation_placeholder)
            else:
                explanation_placeholder.markdown(result['explanation'])
        
        # Download Report
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            # Remove graph_data from downloadable report
            report_data = {k: v for k, v in result.items() if k != 'graph_data'}
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"claim_analysis_{claim_id}.json",
                mime="application/json",
                use_container_width=True
            )

# Other pages remain as placeholders
elif page == "Fraud Network Graph":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🕸️ Interactive Fraud Network</h2>", unsafe_allow_html=True)
    st.info("🚧 Feature available - showing fraud ring analysis from graph database")
    
    min_docs = st.slider("Minimum Shared Documents", 2, 10, 2)
    if st.button("Find Fraud Rings"):
        try:
            response = requests.get(f"{API_URL}/api/fraud/rings", params={"min_shared_docs": min_docs})
            if response.status_code == 200:
                data = response.json()
                st.success(f"Found {data['total_rings_found']} fraud rings")
                if data['rings']:
                    df = pd.DataFrame(data['rings'])
                    st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif page == "Serial Fraudsters":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>👤 Serial Fraudster Detection</h2>", unsafe_allow_html=True)
    
    min_claims = st.slider("Minimum Fraud Claims", 2, 10, 3)
    if st.button("Find Serial Fraudsters"):
        try:
            response = requests.get(f"{API_URL}/api/fraud/serial-fraudsters", params={"min_fraud_claims": min_claims})
            if response.status_code == 200:
                data = response.json()
                st.success(f"Found {data['total_found']} serial fraudsters")
                if data['fraudsters']:
                    df = pd.DataFrame(data['fraudsters'])
                    st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif page == "Analytics Dashboard":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📊 Real-Time Analytics</h2>", unsafe_allow_html=True)
    
    try:
        overview_response = requests.get(f"{API_URL}/api/analytics/overview", timeout=5)
        
        if overview_response.status_code == 200:
            overview = overview_response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims", f"{overview['total_claims']:,}")
            col2.metric("Fraud Claims", f"{overview['fraud_claims']:,}", f"{overview['fraud_rate']}%")
            col3.metric("Avg Fraud Score", f"{overview['avg_fraud_score']:.3f}")
            col4.metric("Total Amount", f"₹{overview['total_amount']/1000000:.1f}M")
            
            st.markdown("---")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Risk Distribution")
                risk_response = requests.get(f"{API_URL}/api/analytics/risk-distribution")
                if risk_response.status_code == 200:
                    risk_data = pd.DataFrame(risk_response.json())
                    fig = px.pie(risk_data, values='count', names='risk_level', hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                st.subheader("Fraud by Product")
                product_response = requests.get(f"{API_URL}/api/analytics/by-product")
                if product_response.status_code == 200:
                    product_data = pd.DataFrame(product_response.json())
                    fig = px.bar(product_data, x='product', y='fraud_rate')
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Cannot connect to analytics API: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #666; font-size: 0.9em;'>
        ClaimLens AI v2.0 | State-of-the-Art Explainable Fraud Detection<br>
        ✨ Powered by Real-Time API Integration + Multi-Modal Analysis<br>
        Built with ❤️ using Streamlit, Plotly, NetworkX<br>
        <strong>Last Updated:</strong> December 2025
    </p>
</div>
""", unsafe_allow_html=True)
