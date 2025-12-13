#!/usr/bin/env python3
"""
ClaimLens v2.0 SOTA Dashboard - Unified API Integration
Now uses complete unified endpoint with ALL modules:
- ML Engine
- CV Engine (Document Verification)
- Graph Engine (Fraud Networks)
- LLM Engine (Semantic Aggregation + Explanations)
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

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ClaimLens AI - Unified Fraud Detection",
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
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);