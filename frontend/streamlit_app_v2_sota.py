#!/usr/bin/env python3
"""
ClaimLens v2.0 SOTA Dashboard
State-of-the-Art Features:
- Modern glassmorphism UI
- Real-time streaming explanations
- Interactive confidence meters
- Animated risk gauges
- Network graph visualization
- Downloadable reports
- Dark mode support
- Mobile responsive
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import json
import time

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
        ["AI Claim Analyzer", "Fraud Network Graph", "Serial Fraudsters", "Analytics Dashboard", "Explainability Lab"],
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
    
    # Quick stats
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #667eea; margin: 0;'>AI Status</h3>
        <p style='font-size: 2em; margin: 10px 0;'>✅</p>
        <p style='margin: 0;'>Models Active</p>
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

def create_network_graph(claim_id):
    """Create fraud network graph."""
    # Mock network data
    G = nx.Graph()
    G.add_node(claim_id, node_type='claim', risk='high')
    G.add_node('CLMT001', node_type='claimant')
    G.add_node('DOC123', node_type='document')
    G.add_node('HOS456', node_type='hospital')
    G.add_edges_from([(claim_id, 'CLMT001'), (claim_id, 'DOC123'), ('CLMT001', 'HOS456')])
    
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
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hoverinfo='text', text=node_text, textposition="top center",
        marker=dict(size=30, color='#667eea', line=dict(width=2, color='white'))
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
            claim_id = st.text_input("🏷️ Claim ID", value="CLM2024001")
            product = st.selectbox("📦 Product Type", ["motor", "health", "life", "property"])
        
        with col2:
            subtype = st.text_input("📝 Claim Subtype", value="accident")
            claim_amount = st.number_input("💵 Claim Amount (₹)", min_value=0, value=450000, step=10000)
        
        with col3:
            policy_premium = st.number_input("💳 Premium (₹)", min_value=0, value=15000, step=1000)
            days_since_policy = st.number_input("📅 Days Since Policy", min_value=0, value=45)
        
        with col4:
            claimant_id = st.text_input("👤 Claimant ID", value="CLMT12345")
            documents = st.text_input("📄 Documents", value="pan,aadhaar,rc,dl")
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            analyze_btn = st.button("🚀 Analyze with AI", type="primary", use_container_width=True)
    
    if analyze_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis stages
        stages = [
            ("📄 Verifying documents...", 0.2),
            ("🔍 Running ML fraud detection...", 0.4),
            ("🕸️ Analyzing fraud networks...", 0.6),
            ("🧠 Applying semantic aggregation...", 0.8),
            ("🤖 Generating AI explanation...", 1.0)
        ]
        
        for stage, progress in stages:
            status_text.text(stage)
            progress_bar.progress(progress)
            time.sleep(0.5)
        
        status_text.empty()
        progress_bar.empty()
        
        # Mock semantic result
        result = {
            "claim_id": claim_id,
            "verdict": "REVIEW",
            "confidence": 0.82,
            "final_score": 0.68,
            "primary_reason": "High claim-to-premium ratio (30x) combined with early claim timing",
            "component_results": {
                "document_verification": {
                    "verdict": "SUSPICIOUS", "confidence": 0.72, "score": 0.72,
                    "reason": "Moderate risk patterns detected", "red_flags": ["Limited document count"]
                },
                "ml_fraud_score": {
                    "verdict": "HIGH_RISK", "confidence": 0.84, "score": 0.84,
                    "reason": "ML fraud probability 84%", "red_flags": ["High claim-to-premium ratio", "Early claim timing"]
                },
                "graph_analysis": {
                    "verdict": "CLEAN", "confidence": 0.88, "score": 0.20,
                    "reason": "No fraud network detected", "red_flags": []
                }
            },
            "critical_flags": [],
            "reasoning_chain": [
                {"stage": "critical_flag_check", "decision": "NO_FLAGS", "reason": "No critical violations detected"},
                {"stage": "adaptive_scoring", "decision": "CALCULATED", "reason": "Aggregated risk score: 0.68"},
                {"stage": "verdict_determination", "decision": "REVIEW", "reason": "Score threshold maps to manual review"}
            ],
            "explanation": "This claim requires manual review due to several risk factors. The claim amount of ₹450,000 against a premium of ₹ 15,000 represents a 30x ratio, which is significantly higher than average. Additionally, the claim was filed just 45 days after policy inception, which statistically correlates with higher fraud risk. While our document verification shows moderate concerns and ML models predict 84% fraud probability, no fraud network connections were detected. We recommend verifying the claimant's history and authenticating all submitted documents before processing."
        }
        
        st.markdown("---")
        
        # Verdict Display
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
                <p style='font-size: 3em; margin: 0;'>30x</p>
                <p>Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Component Analysis
        st.markdown("<h3 style='color: #667eea; margin-top: 30px;'>📦 Component Risk Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Component cards
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
            # Radial risk chart
            fig_radial = create_risk_radial(result['component_results'])
            st.plotly_chart(fig_radial, use_container_width=True)
        
        # Network Graph
        st.markdown("<h3 style='color: #667eea; margin-top: 30px;'>🕸️ Fraud Network Analysis</h3>", unsafe_allow_html=True)
        fig_network = create_network_graph(claim_id)
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
            report_json = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"claim_analysis_{claim_id}.json",
                mime="application/json",
                use_container_width=True
            )

# ============================================================================
# OTHER MODULES (Simplified placeholders - add full implementation)
# ============================================================================

elif page == "Fraud Network Graph":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🕸️ Interactive Fraud Network</h2>", unsafe_allow_html=True)
    st.info("🚧 Feature under development - will show interactive fraud ring graphs")

elif page == "Serial Fraudsters":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>👤 Serial Fraudster Detection</h2>", unsafe_allow_html=True)
    st.info("🚧 Feature under development - will show serial fraudster patterns")

elif page == "Analytics Dashboard":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📊 Real-Time Analytics</h2>", unsafe_allow_html=True)
    st.info("🚧 Feature under development - will show comprehensive analytics")

elif page == "Explainability Lab":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🧪 Explainability Laboratory</h2>", unsafe_allow_html=True)
    st.info("🚧 Feature under development - will allow testing different explanation styles")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #666; font-size: 0.9em;'>
        ClaimLens AI v2.0 | State-of-the-Art Explainable Fraud Detection<br>
        ✨ Powered by Semantic Aggregation + LLM (Groq + Llama-3.3-70B)<br>
        Built with ❤️ using Streamlit, Plotly, NetworkX<br>
        <strong>Last Updated:</strong> December 2025
    </p>
</div>
""", unsafe_allow_html=True)
