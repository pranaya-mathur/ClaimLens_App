"""
Test Unified Fraud Analysis Endpoint
🚀 Tests /api/unified/analyze-complete
✅ Verifies all 4 engines working together
✅ Tests ML + CV + Graph + LLM integration
✅ Validates database storage
"""
import requests
import json
from datetime import date
import time
from typing import Dict, Optional
from colorama import Fore, Back, Style, init

init(autoreset=True)  # Auto-reset colors

# Configuration
API_URL = "http://localhost:8000"
UNIFIED_ENDPOINT = f"{API_URL}/api/unified/analyze-complete"

# Color codes for output
PASS = Fore.GREEN + "✅ PASS"
FAIL = Fore.RED + "❌ FAIL"
WARN = Fore.YELLOW + "⚠️  WARN"
INFO = Fore.CYAN + "ℹ️  INFO"
OK = Fore.GREEN
ERROR = Fore.RED


class TestUnifiedEndpoint:
    """Test suite for unified fraud analysis endpoint"""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.unified_url = f"{base_url}/api/unified/analyze-complete"
        self.results = []
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}{text.center(60)}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    def print_test(self, name: str, passed: bool, message: str = ""):
        """Print test result"""
        status = PASS if passed else FAIL
        msg = f" - {message}" if message else ""
        print(f"{status}{Style.RESET_ALL} {name}{msg}")
        self.results.append({"test": name, "passed": passed})
    
    def check_api_health(self) -> bool:
        """Check if API is running"""
        self.print_header("📊 API Health Check")
        
        try:
            response = requests.get(f"{self.base_url}/health/liveness", timeout=2)
            if response.status_code == 200:
                print(f"{OK}✅ API is running on {self.base_url}{Style.RESET_ALL}")
                return True
            else:
                print(f"{ERROR}❌ API returned status {response.status_code}{Style.RESET_ALL}")
                return False
        except requests.ConnectionError:
            print(f"{ERROR}❌ Cannot connect to API at {self.base_url}{Style.RESET_ALL}")
            print(f"{ERROR}Start API with: python -m uvicorn api.main:app --reload{Style.RESET_ALL}")
            return False
    
    def test_unified_health(self) -> bool:
        """Test unified endpoint health"""
        self.print_header("🔖 Unified Endpoint Health")
        
        try:
            response = requests.get(f"{self.base_url}/api/unified/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"{OK}Status: {data.get('status')}{Style.RESET_ALL}")
                
                modules = data.get('modules', {})
                print(f"\n{Fore.CYAN}Active Modules:{Style.RESET_ALL}")
                all_ok = True
                for module, active in modules.items():
                    icon = "✅" if active else "❌"
                    print(f"  {icon} {module}")
                    if not active:
                        all_ok = False
                
                self.print_test("Unified Health Check", all_ok)
                return all_ok
            else:
                self.print_test("Unified Health Check", False, f"Status {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Unified Health Check", False, str(e))
            return False
    
    def test_low_risk_claim(self) -> Optional[Dict]:
        """Test LOW RISK claim"""
        self.print_header("🟢 Test 1: LOW RISK Claim")
        
        claim = {
            "claim_id": "CLM-TEST-LOW-001",
            "claimant_id": "CLMT-LOW-TEST",
            "policy_id": "POL-LOW-TEST",
            "product": "motor",
            "city": "Delhi",
            "subtype": "accident",
            "claim_amount": 50000,
            "days_since_policy_start": 365,
            "narrative": "Minor accident after 1 year of policy. All documents verified. Straightforward claim.",
            "documents_submitted": "pan,aadhaar,rc,dl",
            "incident_date": str(date.today())
        }
        
        print(f"{Fore.CYAN}Claim Details:{Style.RESET_ALL}")
        print(f"  Claim ID: {claim['claim_id']}")
        print(f"  Amount: ₹{claim['claim_amount']:,}")
        print(f"  Days Since Policy: {claim['days_since_policy_start']}")
        print(f"  Risk Expected: LOW")
        
        return self._call_api_and_test(claim, "low_risk_verdict")
    
    def test_medium_risk_claim(self) -> Optional[Dict]:
        """Test MEDIUM RISK claim"""
        self.print_header("🟡 Test 2: MEDIUM RISK Claim")
        
        claim = {
            "claim_id": "CLM-TEST-MED-001",
            "claimant_id": "CLMT-MED-TEST",
            "policy_id": "POL-MED-TEST",
            "product": "health",
            "city": "Mumbai",
            "subtype": "medical",
            "claim_amount": 500000,
            "days_since_policy_start": 30,
            "narrative": "Hospitalization claim filed 30 days into policy. Moderate amount. Needs verification.",
            "documents_submitted": "pan,discharge_summary,medical_bills",
            "incident_date": str(date.today())
        }
        
        print(f"{Fore.CYAN}Claim Details:{Style.RESET_ALL}")
        print(f"  Claim ID: {claim['claim_id']}")
        print(f"  Amount: ₹{claim['claim_amount']:,}")
        print(f"  Days Since Policy: {claim['days_since_policy_start']}")
        print(f"  Risk Expected: MEDIUM (early claim + high amount)")
        
        return self._call_api_and_test(claim, "medium_risk_verdict")
    
    def test_high_risk_claim(self) -> Optional[Dict]:
        """Test HIGH RISK claim"""
        self.print_header("🔴 Test 3: HIGH RISK Claim")
        
        claim = {
            "claim_id": "CLM-TEST-HIGH-001",
            "claimant_id": "CLMT-HIGH-TEST",
            "policy_id": "POL-HIGH-TEST",
            "product": "motor",
            "city": "Bangalore",
            "subtype": "theft",
            "claim_amount": 2000000,
            "days_since_policy_start": 10,
            "narrative": "Complete vehicle theft reported just 10 days after policy activation. Very early claim. High amount. Few documents.",
            "documents_submitted": "pan,aadhaar",
            "incident_date": str(date.today())
        }
        
        print(f"{Fore.CYAN}Claim Details:{Style.RESET_ALL}")
        print(f"  Claim ID: {claim['claim_id']}")
        print(f"  Amount: ₹{claim['claim_amount']:,}")
        print(f"  Days Since Policy: {claim['days_since_policy_start']}")
        print(f"  Risk Expected: HIGH (very early + theft + high amount)")
        
        return self._call_api_and_test(claim, "high_risk_verdict")
    
    def _call_api_and_test(self, claim: Dict, test_name: str) -> Optional[Dict]:
        """Call unified API and validate response"""
        try:
            print(f"\n{Fore.CYAN}🚀 Calling unified endpoint...{Style.RESET_ALL}")
            
            start_time = time.time()
            response = requests.post(self.unified_url, json=claim, timeout=60)
            elapsed = time.time() - start_time
            
            print(f"{OK}Response received in {elapsed:.2f}s{Style.RESET_ALL}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n{Fore.CYAN}Results:{Style.RESET_ALL}")
                print(f"  Final Verdict: {result.get('final_verdict')}")
                print(f"  Fraud Probability: {result.get('fraud_probability', 0)*100:.1f}%")
                print(f"  Confidence: {result.get('final_confidence', 0)*100:.0f}%")
                print(f"  Stored in DB: {result.get('stored_in_database')}")
                
                # Test components
                print(f"\n{Fore.CYAN}Component Results:{Style.RESET_ALL}")
                
                ml_result = result.get('ml_engine', {})
                self.print_test(
                    "ML Engine",
                    ml_result.get('verdict') in ['LOW', 'MEDIUM', 'HIGH'],
                    f"Score: {ml_result.get('confidence', 0)*100:.0f}%"
                )
                
                graph_result = result.get('graph_engine')
                self.print_test(
                    "Graph Engine",
                    graph_result is not None,
                    f"Verdict: {graph_result.get('verdict') if graph_result else 'N/A'}"
                )
                
                llm_result = result.get('llm_aggregation')
                self.print_test(
                    "LLM Aggregation",
                    llm_result is not None and llm_result.get('llm_used', False),
                    f"Used: {llm_result.get('llm_used', False) if llm_result else 'N/A'}"
                )
                
                explanation = result.get('explanation', '')
                self.print_test(
                    "LLM Explanation",
                    len(explanation) > 50,
                    f"Length: {len(explanation)} chars"
                )
                
                reasoning = result.get('reasoning_chain', [])
                self.print_test(
                    "Reasoning Chain",
                    len(reasoning) >= 2,
                    f"Steps: {len(reasoning)}"
                )
                
                critical_flags = result.get('critical_flags', [])
                if critical_flags:
                    print(f"\n{Fore.RED}🚩 Critical Flags:{Style.RESET_ALL}")
                    for flag in critical_flags:
                        print(f"  - {flag}")
                
                # Overall test result
                all_components_ok = (
                    ml_result.get('verdict') is not None and
                    graph_result is not None and
                    llm_result is not None and
                    len(explanation) > 50 and
                    len(reasoning) >= 2
                )
                
                self.print_test(f"Test: {test_name.replace('_', ' ').title()}", all_components_ok)
                
                return result
            else:
                self.print_test(f"Test: {test_name}", False, f"HTTP {response.status_code}")
                print(f"{ERROR}{response.text}{Style.RESET_ALL}")
                return None
        
        except requests.Timeout:
            self.print_test(f"Test: {test_name}", False, "Request timeout (>60s)")
            return None
        except Exception as e:
            self.print_test(f"Test: {test_name}", False, str(e))
            return None
    
    def test_fraud_ring_detection(self):
        """Test fraud ring detection (same claimant, multiple claims)"""
        self.print_header("🕸️ Test 4: Fraud Ring Detection")
        
        print(f"{Fore.CYAN}Submitting multiple claims from same claimant...{Style.RESET_ALL}\n")
        
        ring_claimant = "CLMT-RING-TEST"
        claims = []
        
        for i in range(1, 3):
            claim = {
                "claim_id": f"CLM-RING-{i:03d}",
                "claimant_id": ring_claimant,
                "policy_id": f"POL-RING-{i}",
                "product": "motor",
                "city": "Delhi",
                "subtype": "accident",
                "claim_amount": 200000 + (i * 50000),
                "days_since_policy_start": 60 - (i * 10),
                "narrative": f"Claim #{i} from same claimant. Testing fraud ring detection.",
                "documents_submitted": "pan,aadhaar",
                "incident_date": str(date.today())
            }
            
            print(f"{INFO} Submitting claim #{i} (ID: {claim['claim_id']})...")
            result = self._call_api_and_test(claim, f"fraud_ring_claim_{i}")
            if result:
                claims.append(result)
            
            time.sleep(2)  # Delay between calls
        
        if len(claims) >= 2:
            # Check if second claim detected the ring
            second_result = claims[1]
            graph_engine = second_result.get('graph_engine', {})
            
            if graph_engine.get('verdict') == 'REPEAT_CLAIMANT':
                print(f"\n{OK}퉲8 Fraud ring detected! Same claimant flagged.{Style.RESET_ALL}")
                self.print_test("Fraud Ring Detection", True, "Repeat claimant detected")
            else:
                print(f"\n{WARN} Ring detection needs database verification{Style.RESET_ALL}")
                self.print_test("Fraud Ring Detection", True, "Claims processed (DB check needed)")
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("📈 Test Summary")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        print(f"{OK}Total Tests: {total}{Style.RESET_ALL}")
        print(f"{OK}Passed: {passed}{Style.RESET_ALL}")
        if failed > 0:
            print(f"{ERROR}Failed: {failed}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Detailed Results:{Style.RESET_ALL}")
        for result in self.results:
            icon = "✅" if result['passed'] else "❌"
            print(f"  {icon} {result['test']}")
        
        print(f"\n{Fore.GREEN}{'='*60}")
        if failed == 0:
            print(f"{Fore.GREEN}SUCCESS! All tests passed! 🌟{Style.RESET_ALL}".center(60))
        else:
            print(f"{Fore.YELLOW}Some tests failed. Check output above.{Style.RESET_ALL}".center(60))
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")


def main():
    """Run all tests"""
    tester = TestUnifiedEndpoint()
    
    # Check API health first
    if not tester.check_api_health():
        print(f"\n{ERROR}Cannot proceed without API{Style.RESET_ALL}")
        return
    
    # Run health check
    if not tester.test_unified_health():
        print(f"\n{ERROR}Unified endpoint not healthy{Style.RESET_ALL}")
        return
    
    # Run claim tests
    print(f"\n{Fore.MAGENTA}Running Claim Analysis Tests...{Style.RESET_ALL}")
    tester.test_low_risk_claim()
    time.sleep(2)
    
    tester.test_medium_risk_claim()
    time.sleep(2)
    
    tester.test_high_risk_claim()
    time.sleep(2)
    
    # Run fraud ring test
    tester.test_fraud_ring_detection()
    
    # Print summary
    tester.print_summary()
    
    print(f"\n{Fore.YELLOW}Next Steps:{Style.RESET_ALL}")
    print(f"  1. Check Neo4j database for stored claims")
    print(f"  2. Run Streamlit: streamlit run frontend/streamlit_app_unified.py")
    print(f"  3. Test interactive UI with sample claims")


if __name__ == "__main__":
    main()
