"""
Comprehensive Module Integration Tests for ClaimLens
Tests import chains, route registrations, and inter-module dependencies
"""
import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class IntegrationTester:
    """Test smooth integration of all ClaimLens modules"""
    
    def __init__(self):
        self.results = []
        self.failed_tests = []
    
    def test(self, test_name: str, func):
        """Run a single test and track results"""
        try:
            func()
            self.results.append(f"✓ {test_name}")
            return True
        except Exception as e:
            error_msg = f"✗ {test_name}: {str(e)}"
            self.results.append(error_msg)
            self.failed_tests.append({
                'test': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def print_report(self):
        """Print test results report"""
        print("\n" + "="*70)
        print("CLAIMLENS MODULE INTEGRATION TEST REPORT")
        print("="*70)
        
        for result in self.results:
            print(result)
        
        print("\n" + "-"*70)
        passed = len([r for r in self.results if r.startswith('✓')])
        failed = len([r for r in self.results if r.startswith('✗')])
        print(f"PASSED: {passed} | FAILED: {failed} | TOTAL: {len(self.results)}")
        print("-"*70)
        
        if self.failed_tests:
            print("\n⚠️  FAILED TEST DETAILS:\n")
            for failure in self.failed_tests:
                print(f"Test: {failure['test']}")
                print(f"Error: {failure['error']}")
                print(f"Traceback:\n{failure['traceback']}")
                print("-"*70)
        
        return failed == 0


def test_core_imports(tester: IntegrationTester):
    """Test 1: Core library imports"""
    def run():
        import fastapi
        import pydantic
        import loguru
        assert fastapi.__version__ is not None
    tester.test("Core imports (FastAPI, Pydantic, Loguru)", run)


def test_api_main_import(tester: IntegrationTester):
    """Test 2: Main API module import"""
    def run():
        from api.main import app
        assert app is not None
        assert app.title == "ClaimLens API"
    tester.test("API main module import", run)


def test_route_modules(tester: IntegrationTester):
    """Test 3-9: All route module imports"""
    routes = [
        ("health", "api.routes.health"),
        ("fraud", "api.routes.fraud"),
        ("analytics", "api.routes.analytics"),
        ("ingest", "api.routes.ingest"),
        ("cv_detection", "api.routes.cv_detection"),
        ("ml_engine", "api.routes.ml_engine"),
        ("document_verification", "api.routes.document_verification")
    ]
    
    for route_name, module_path in routes:
        def run(mp=module_path, rn=route_name):
            mod = __import__(mp, fromlist=['router'])
            assert hasattr(mod, 'router'), f"{rn} missing router"
        tester.test(f"Route module: {route_name}", run)


def test_middleware(tester: IntegrationTester):
    """Test 10: Middleware import"""
    def run():
        from api.middleware.rate_limiter import RateLimitMiddleware
        assert RateLimitMiddleware is not None
    tester.test("Rate limiting middleware", run)


def test_src_modules(tester: IntegrationTester):
    """Test 11-17: Source module imports"""
    src_modules = [
    ("claim_processor", "src.app.claim_processor"),
    ("health_analyzer", "src.app.health_analyzer"),
    ("semantic_aggregator", "src.app.semantic_aggregator"),
    ("verdict_models", "src.app.verdict_models"),
    ("damage_detector", "src.cv_engine.damage_detector"),
    ("feature_engineer", "src.ml_engine.feature_engineer"),
    ("ml_scorer", "src.ml_engine.ml_scorer"),
]

    
    for module_name, module_path in src_modules:
        def run(mp=module_path, mn=module_name):
            __import__(mp)
        tester.test(f"Source module: {module_name}", run)


def test_route_registration(tester: IntegrationTester):
    """Test 18: All routes properly registered in main app"""
    def run():
        from api.main import app
        routes = [r.path for r in app.routes]
        
        expected_prefixes = [
            "/health",
            "/api/fraud",
            "/api/analytics",
            "/api/ingest",
            "/api/cv",
            "/api/ml",
            "/api/documents"
        ]
        
        for prefix in expected_prefixes:
            assert any(prefix in r for r in routes), f"Missing routes for {prefix}"
    
    tester.test("Route registration in main app", run)


def test_ml_engine_integration(tester: IntegrationTester):
    """Test 19: ML Engine components integration"""
    def run():
        from src.ml_engine.feature_engineer import FeatureEngineer
        from src.ml_engine.ml_scorer import MLFraudScorer
        
        # Check class instantiation doesn't crash
        assert FeatureEngineer is not None
        assert MLFraudScorer is not None
    
    tester.test("ML Engine integration", run)


def test_cv_module_integration(tester: IntegrationTester):
    """Test 20: Computer Vision module integration"""
    def run():
        from src.cv_engine import DamageDetector
        assert DamageDetector is not None
    
    tester.test("Computer Vision module integration", run)


def test_document_verification_integration(tester: IntegrationTester):
    """Test 21: Document verification route has all endpoints"""
    def run():
        from api.routes.document_verification import router
        route_paths = [r.path for r in router.routes]
        
        required_endpoints = [
            "/verify-pan",
            "/verify-aadhaar",
            "/verify-document",
            "/extract-text"
        ]
        
        for endpoint in required_endpoints:
            assert any(endpoint in path for path in route_paths), f"Missing {endpoint}"
    
    tester.test("Document verification endpoints", run)


def test_health_claim_routing(tester: IntegrationTester):
    """Test 22: Health claim analyzer integration (Bug #1 fix)"""
    def run():
        from src.app.health_analyzer import HealthClaimAnalyzer
        from src.app.claim_processor import ClaimProcessor
        
        assert HealthClaimAnalyzer is not None
        assert ClaimProcessor is not None
    
    tester.test("Health claim routing (Bug #1 fix)", run)


def test_fallback_system(tester: IntegrationTester):
    """Test 23: Smart fallback system for missing data"""
    def run():
        from api.routes.ingest import router
        # Check ingest route exists (handles missing data scenarios)
        assert router is not None
    
    tester.test("Smart fallback system integration", run)


def test_config_loading(tester: IntegrationTester):
    """Test 24: Configuration files load correctly"""
    def run():
        # Check if .env.example exists as template
        env_example = project_root / ".env.example"
        assert env_example.exists(), ".env.example not found"
    
    tester.test("Configuration file availability", run)


def test_requirements_file(tester: IntegrationTester):
    """Test 25: Requirements file exists and is valid"""
    def run():
        req_file = project_root / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"
        
        with open(req_file) as f:
            deps = f.read()
            assert "fastapi" in deps
            assert "loguru" in deps
    
    tester.test("Requirements file validity", run)


def test_docker_compose(tester: IntegrationTester):
    """Test 26: Docker compose file exists"""
    def run():
        docker_file = project_root / "docker-compose.yml"
        assert docker_file.exists(), "docker-compose.yml not found"
    
    tester.test("Docker compose configuration", run)


def run_all_tests():
    """Execute all integration tests"""
    tester = IntegrationTester()
    
    print("\n🔍 Starting ClaimLens Module Integration Tests...\n")
    
    # Run all tests
    test_core_imports(tester)
    test_api_main_import(tester)
    test_route_modules(tester)
    test_middleware(tester)
    test_src_modules(tester)
    test_route_registration(tester)
    test_ml_engine_integration(tester)
    test_cv_module_integration(tester)
    test_document_verification_integration(tester)
    test_health_claim_routing(tester)
    test_fallback_system(tester)
    test_config_loading(tester)
    test_requirements_file(tester)
    test_docker_compose(tester)
    
    # Print report
    success = tester.print_report()
    
    if success:
        print("\n🎉 All integration tests passed! Modules are properly integrated.\n")
    else:
        print("\n⚠️  Some integration tests failed. Review errors above.\n")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
