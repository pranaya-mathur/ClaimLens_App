# ClaimLens Integration Testing Guide

## Overview

This guide helps you verify that all ClaimLens modules integrate smoothly and can run together without conflicts. The integration test suite checks:

- All module imports work correctly
- Route registrations are complete
- Dependencies are satisfied
- Bug fixes remain stable
- ML Engine, CV, and Document Verification components load properly

---

## Quick Start

### 1. Run Integration Tests

```bash
# From project root
python tests/integration/test_module_integration.py
```

### 2. Expected Output

You should see:

```
Starting ClaimLens Module Integration Tests...

======================================================================
CLAIMLENS MODULE INTEGRATION TEST REPORT
======================================================================
PASSED: Core imports (FastAPI, Pydantic, Loguru)
PASSED: API main module import
PASSED: Route module: health
PASSED: Route module: fraud
PASSED: Route module: analytics
PASSED: Route module: ingest
PASSED: Route module: cv_detection
PASSED: Route module: ml_engine
PASSED: Route module: document_verification
PASSED: Rate limiting middleware
PASSED: Source module: claim_processor
PASSED: Source module: motor_analyzer
PASSED: Source module: health_analyzer
PASSED: Source module: risk_assessor
PASSED: Source module: cv_detector
PASSED: Source module: feature_engineer
PASSED: Source module: fraud_scorer
PASSED: Route registration in main app
PASSED: ML Engine integration
PASSED: Computer Vision module integration
PASSED: Document verification endpoints
PASSED: Health claim routing (Bug #1 fix)
PASSED: Smart fallback system integration
PASSED: Configuration file availability
PASSED: Requirements file validity
PASSED: Docker compose configuration

----------------------------------------------------------------------
PASSED: 26 | FAILED: 0 | TOTAL: 26
----------------------------------------------------------------------

All integration tests passed! Modules are properly integrated.
```

---

## What Gets Tested

### Module Import Tests (Tests 1-17)

| Test # | Component | What's Checked |
|--------|-----------|----------------|
| 1 | Core libraries | FastAPI, Pydantic, Loguru load |
| 2 | Main API | `api.main` imports and app initializes |
| 3-9 | Route modules | All 7 route files import correctly |
| 10 | Middleware | Rate limiter loads |
| 11-17 | Source modules | Claim processors, analyzers, ML engine, CV detector |

### Integration Tests (Tests 18-26)

| Test # | Integration Check | Purpose |
|--------|-------------------|----------|
| 18 | Route registration | All 7 route prefixes registered in app |
| 19 | ML Engine | FeatureEngineer + FraudScorer integration |
| 20 | Computer Vision | DamageDetector integration |
| 21 | Document verification | All 4 doc endpoints present |
| 22 | Health claim routing | Bug #1 fix verification |
| 23 | Fallback system | Missing data handling |
| 24 | Config loading | .env.example exists |
| 25 | Requirements | Dependencies file valid |
| 26 | Docker setup | docker-compose.yml exists |

---

## Running Specific Test Categories

### Test Individual Modules

```python
# In Python shell or notebook
from tests.integration.test_module_integration import IntegrationTester, test_route_modules

tester = IntegrationTester()
test_route_modules(tester)
tester.print_report()
```

### Test Only ML Engine

```python
from tests.integration.test_module_integration import IntegrationTester, test_ml_engine_integration

tester = IntegrationTester()
test_ml_engine_integration(tester)
tester.print_report()
```

---

## Troubleshooting Common Issues

### ImportError: No module named 'api'

**Cause**: Python can't find the project root.

**Fix**:
```bash
# Set PYTHONPATH before running
export PYTHONPATH="$PWD:$PYTHONPATH"
python tests/integration/test_module_integration.py
```

### Missing dependencies

**Cause**: Some required packages aren't installed.

**Fix**:
```bash
pip install -r requirements.txt
```

### Route registration test fails

**Cause**: A route module wasn't imported in `api/main.py`.

**Fix**: Check that `api/main.py` includes:
```python
from api.routes import fraud, health, analytics, ingest, cv_detection, ml_engine, document_verification

app.include_router(fraud.router, prefix="/api/fraud", tags=["Fraud Detection"])
# ... etc for all routes
```

### ML Engine test fails

**Cause**: ML model files or embeddings missing.

**Check**:
- `models/` directory has trained model files
- Environment variables set for model paths (if using)

---

## Integration with CI/CD

### GitHub Actions Workflow

Add to `.github/workflows/integration-tests.yml`:

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: python tests/integration/test_module_integration.py
```

### Docker Testing

```bash
# Build and test in container
docker-compose build
docker-compose run --rm api python tests/integration/test_module_integration.py
```

---

## Pre-Deployment Checklist

Before deploying ClaimLens, verify:

- [ ] All 26 integration tests pass
- [ ] No import errors in any module
- [ ] All routes accessible at documented paths
- [ ] ML models load without errors
- [ ] Document verification endpoints respond
- [ ] Rate limiting middleware active
- [ ] Environment variables configured (`.env` from `.env.example`)
- [ ] Docker container builds successfully

---

## Adding New Tests

When adding new modules or routes:

1. **Add import test**:
```python
def test_new_module(tester: IntegrationTester):
    def run():
        from api.routes.new_module import router
        assert router is not None
    tester.test("New module import", run)
```

2. **Add to `run_all_tests()`**:
```python
def run_all_tests():
    tester = IntegrationTester()
    # ... existing tests ...
    test_new_module(tester)
    # ...
```

3. **Update route registration check**:
Add your new route prefix to `expected_prefixes` in `test_route_registration()`.

---

## Contact & Support

If integration tests fail unexpectedly:
1. Check the detailed error traceback in the test output
2. Verify all dependencies are installed: `pip list`
3. Ensure you're on the correct branch: `git branch`
4. Review recent commits for breaking changes

**Test Coverage**: 26 integration points across 7 API routes, 7 source modules, ML engine, CV system, and document verification.

---

## Version History

- **v1.0** (Dec 12, 2025): Initial integration test suite covering all modules and bug fixes #1-#10
