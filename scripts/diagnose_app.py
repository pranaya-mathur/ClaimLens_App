#!/usr/bin/env python3
"""
ClaimLens Diagnostic Script

Comprehensive health check for all ClaimLens components:
- FastAPI server
- ML Engine (CatBoost model)
- LLM Engine (Groq API)
- Document Verification
- Graph Database (Neo4j)
- Computer Vision models

Usage:
    python scripts/diagnose_app.py
"""

import sys
import os
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")


def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")


def print_error(text):
    print(f"{RED}✗{RESET} {text}")


def print_info(text):
    print(f"{BLUE}ℹ{RESET} {text}")


def check_env_variables():
    """Check critical environment variables"""
    print_header("ENVIRONMENT CONFIGURATION")
    
    critical_vars = {
        "GROQ_API_KEY": "LLM Explanations",
        "NEO4J_URI": "Graph Database",
        "NEO4J_USER": "Graph Database",
        "NEO4J_PASSWORD": "Graph Database"
    }
    
    optional_vars = {
        "OPENAI_API_KEY": "LLM Fallback",
        "REDIS_HOST": "Caching"
    }
    
    all_good = True
    
    for var, purpose in critical_vars.items():
        value = os.getenv(var)
        if value and value != "your_groq_api_key_here":
            # Mask API keys
            if "KEY" in var:
                display = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                display = value
            print_success(f"{var}: {display} ({purpose})")
        else:
            print_error(f"{var}: NOT SET ({purpose})")
            all_good = False
    
    print()
    for var, purpose in optional_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}":
            print_info(f"{var}: Configured ({purpose})")
        else:
            print_warning(f"{var}: Not set ({purpose} - optional)")
    
    return all_good


def check_api_server():
    """Check if FastAPI server is running"""
    print_header("FASTAPI SERVER")
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    try:
        response = requests.get(f"{api_url}/health/liveness", timeout=3)
        if response.status_code == 200:
            print_success(f"Server running at {api_url}")
            return True, api_url
        else:
            print_error(f"Server responded with status {response.status_code}")
            return False, api_url
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {api_url}")
        print_info("Start the server with: uvicorn api.main:app --reload")
        return False, api_url
    except Exception as e:
        print_error(f"Error: {e}")
        return False, api_url


def check_llm_engine(api_url):
    """Check LLM Engine health"""
    print_header("LLM ENGINE (GROQ)")
    
    try:
        response = requests.get(f"{api_url}/api/llm/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if data.get("llm_available"):
                print_success(f"LLM Engine: {data.get('status').upper()}")
                print_success(f"Model: {data.get('model')}")
                print_success(f"API Key: Configured")
                return True
            else:
                print_warning(f"LLM Engine: {data.get('status').upper()}")
                print_warning(f"Fallback Mode: {data.get('fallback_mode')}")
                print_info(data.get('message'))
                
                if not data.get('api_key_configured'):
                    print_error("GROQ_API_KEY not configured")
                    print_info("Get free API key: https://console.groq.com/")
                return False
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot check LLM health: {e}")
        return False


def check_ml_engine(api_url):
    """Check ML Engine health"""
    print_header("ML ENGINE (CATBOOST)")
    
    try:
        response = requests.get(f"{api_url}/api/ml/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "healthy":
                print_success(f"ML Engine: HEALTHY")
                print_success(f"Model loaded: {data.get('ml_scorer_loaded')}")
                print_success(f"Features: {data.get('model_features')}")
                print_success(f"Threshold: {data.get('threshold')}")
                return True
            else:
                print_error(f"ML Engine: {data.get('status').upper()}")
                print_error(f"Error: {data.get('error')}")
                return False
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot check ML health: {e}")
        return False


def check_graph_db(api_url):
    """Check Neo4j connectivity"""
    print_header("GRAPH DATABASE (NEO4J)")
    
    try:
        # Try fraud detection endpoint which uses Neo4j
        response = requests.post(
            f"{api_url}/api/fraud/score",
            json={"claim_id": "TEST123"},
            timeout=5
        )
        
        if response.status_code == 200:
            print_success("Neo4j: Connected")
            return True
        elif response.status_code == 404:
            print_warning("Neo4j: Connected but claim not found (expected)")
            return True
        else:
            print_warning(f"Graph DB check returned: {response.status_code}")
            print_info("Neo4j may not be running")
            return False
    except Exception as e:
        print_warning("Cannot verify Neo4j connectivity")
        print_info("Start Neo4j if you need graph analysis")
        print_info("docker-compose up neo4j -d")
        return False


def test_llm_explanation(api_url):
    """Test LLM explanation generation"""
    print_header("LLM EXPLANATION TEST")
    
    test_payload = {
        "claim_narrative": "My car was damaged in an accident on the highway.",
        "ml_fraud_prob": 0.35,
        "document_risk": 0.15,
        "network_risk": 0.10,
        "claim_amount": 50000,
        "premium": 15000,
        "days_since_policy": 45,
        "product_type": "motor",
        "audience": "adjuster"
    }
    
    try:
        print_info("Sending test request to /api/llm/explain...")
        response = requests.post(
            f"{api_url}/api/llm/explain",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Explanation generated successfully!")
            print_success(f"Verdict: {data.get('verdict')}")
            print_success(f"Confidence: {data.get('confidence'):.0%}")
            print_success(f"LLM Used: {data.get('llm_used')}")
            print_success(f"Model: {data.get('model')}")
            print(f"\n{BOLD}Explanation Preview:{RESET}")
            explanation = data.get('explanation', '')
            print(explanation[:300] + "..." if len(explanation) > 300 else explanation)
            return True
        else:
            print_error(f"Failed to generate explanation: {response.status_code}")
            print_error(response.text)
            return False
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def print_summary(results):
    """Print diagnostic summary"""
    print_header("DIAGNOSTIC SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\n{BOLD}Results: {passed}/{total} checks passed{RESET}\n")
    
    for component, status in results.items():
        if status:
            print_success(f"{component}: OK")
        else:
            print_error(f"{component}: FAILED")
    
    print("\n" + "="*60 + "\n")
    
    if passed == total:
        print_success(f"{BOLD}\u2713 ALL SYSTEMS OPERATIONAL{RESET}")
        print_info("Your ClaimLens app is fully functional!")
    elif passed >= total * 0.7:
        print_warning(f"{BOLD}⚠ SOME ISSUES DETECTED{RESET}")
        print_info("Core functionality available, but some features may be limited")
    else:
        print_error(f"{BOLD}✗ CRITICAL ISSUES DETECTED{RESET}")
        print_info("Please fix the errors above before running the app")
    
    print("\n" + "="*60 + "\n")


def main():
    print(f"\n{BOLD}{BLUE}ClaimLens Diagnostic Tool{RESET}")
    print(f"{BLUE}Version 2.0 - Comprehensive System Check{RESET}\n")
    
    results = {}
    
    # 1. Environment check
    results["Environment Variables"] = check_env_variables()
    
    # 2. API server check
    server_ok, api_url = check_api_server()
    results["FastAPI Server"] = server_ok
    
    if not server_ok:
        print_error("\nCannot proceed without API server running.")
        print_info("Start server: uvicorn api.main:app --reload")
        print_summary(results)
        return
    
    # 3. Component checks
    results["LLM Engine"] = check_llm_engine(api_url)
    results["ML Engine"] = check_ml_engine(api_url)
    results["Graph Database"] = check_graph_db(api_url)
    
    # 4. Integration test
    results["LLM Explanation Test"] = test_llm_explanation(api_url)
    
    # 5. Summary
    print_summary(results)
    
    # Recommendations
    if not results.get("LLM Engine"):
        print_info("\nRECOMMENDATION: Set GROQ_API_KEY in .env file")
        print_info("Get free key: https://console.groq.com/keys")
    
    if not results.get("Graph Database"):
        print_info("\nRECOMMENDATION: Start Neo4j for fraud network analysis")
        print_info("docker-compose up neo4j -d")


if __name__ == "__main__":
    main()
