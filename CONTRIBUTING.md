# Contributing to ClaimLens AI

Thank you for your interest in contributing to ClaimLens AI!

We welcome contributions from the community, whether it's bug reports, feature requests, documentation improvements, or code contributions.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (coming soon). Please be respectful and constructive in all interactions.

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** (coming soon)
3. **Include details**:
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs if applicable
   - Environment (OS, Python version, etc.)

### Suggesting Features

1. **Search existing feature requests** first
2. **Describe the feature** clearly
3. **Explain the use case** and benefits
4. **Consider implementation** if possible

### Contributing Code

We welcome:
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test coverage improvements

---

## Development Setup

### Prerequisites

```bash
# Required
Python 3.9+
Git

# Optional (for full functionality)
Docker
Neo4j (for graph engine)
```

### Clone and Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ClaimLens_App.git
cd ClaimLens_App

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 6. Download models (see models/README.md)
# Place models in respective folders

# 7. Run tests
pytest tests/

# 8. Start development server
uvicorn api.main:app --reload
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Create feature branch from main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new features
- Update documentation if needed

### 3. Commit Your Changes

Use conventional commit messages:

```bash
# Format: <type>(<scope>): <subject>

# Examples:
git commit -m "feat(ml): Add new fraud detection feature"
git commit -m "fix(cv): Resolve document parsing bug"
git commit -m "docs: Update installation guide"
git commit -m "test: Add unit tests for ML engine"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Use the PR template (coming soon)
```

### 5. PR Review Process

- Maintainers will review your PR
- Address feedback and comments
- Once approved, PR will be merged
- Congratulations, you're a contributor!

---

## Coding Standards

### Python Style

- Follow **PEP 8** guidelines
- Use **type hints** for function parameters and returns
- Keep functions **small and focused**
- Add **docstrings** for classes and functions

```python
def calculate_fraud_score(claim_data: dict, config: Config) -> float:
    """
    Calculate fraud risk score for a claim.
    
    Args:
        claim_data: Dictionary containing claim information
        config: Configuration object with model settings
    
    Returns:
        float: Fraud probability between 0.0 and 1.0
    
    Raises:
        ValueError: If claim_data is invalid
    """
    pass
```

### Code Organization

```
src/
├── ml/          # ML engine
├── cv/          # Computer vision
├── graph/       # Graph analysis
├── llm/         # LLM integration
└── utils/       # Shared utilities
```

---

## Testing Guidelines

### Run Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_ml_engine.py

# With coverage
pytest --cov=src tests/
```

### Write Tests

```python
import pytest
from src.ml.fraud_detector import FraudDetector

def test_fraud_detection():
    """Test basic fraud detection."""
    detector = FraudDetector()
    result = detector.predict(sample_claim)
    
    assert 0.0 <= result['fraud_probability'] <= 1.0
    assert 'risk_level' in result
    assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
```

---

## Documentation

### Update Documentation

When adding features:
- Update relevant `docs/` files
- Add examples to `examples/`
- Update API documentation
- Update CHANGELOG.md

### Documentation Style

- Use clear, concise language
- Add code examples
- Include screenshots for UI changes
- Update table of contents

---

## Areas for Contribution

### High Priority
- Unit test coverage improvement
- Performance optimization
- Documentation enhancements
- Bug fixes

### Feature Ideas
- Batch processing API
- PDF document support
- Multi-language support
- Real-time monitoring dashboard
- Enhanced visualization

---

## Questions?

If you have questions:
- Check [Documentation](docs/README.md)
- Open a [Discussion](https://github.com/pranaya-mathur/ClaimLens_App/discussions)
- Email: pranaya.mathur@yahoo.com

---

## Thank You!

Your contributions make ClaimLens AI better for everyone. We appreciate your time and effort!

---

**Happy Coding!**
