# Contributing to nanocortex

Thank you for your interest in contributing to nanocortex! This document provides guidelines for contributing to the project.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/STiFLeR7/nanocortex.git
cd nanocortex

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # macOS/Linux

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nanocortex --cov-report=term-missing
```

### 4. Lint & Format

```bash
# Check linting
ruff check src/

# Auto-fix issues
ruff check src/ --fix

# Type checking
mypy src/nanocortex/
```

### 5. Submit a Pull Request

- Write a clear PR description
- Reference any related issues
- Ensure all tests pass

---

## 🏗️ Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/nanocortex/` | Core library code |
| `tests/` | pytest test suite |
| `scripts/` | Demo and utility scripts |
| `examples/` | Real-world usage examples |

---

## 📚 Code Style Guidelines

### Python

- Use type hints for all function signatures
- Use Pydantic models for data structures
- Follow PEP 8 naming conventions
- Keep functions focused and testable

### Documentation

- Add docstrings to all public functions/classes
- Update README.md for user-facing changes
- Include examples in docstrings where helpful

---

## 🧪 Testing Guidelines

- Write unit tests for individual components
- Write integration tests for cross-layer functionality
- Use fixtures from `tests/conftest.py`
- Mark integration tests with `@pytest.mark.integration`

---

## 📝 Commit Messages

Use clear, descriptive commit messages:

```
feat: Add vector retrieval to knowledge layer
fix: Resolve document close timing issue in perception
docs: Update README with API endpoint examples
test: Add unit tests for policy engine conditions
```

---

## 🔗 Useful Links

- [README](./README.md) - Project overview and setup
- [LICENSE](./LICENSE) - MIT License
- [API Docs](http://localhost:8000/docs) - Interactive Swagger (when running)

---

## ❓ Questions?

Open an issue on GitHub or reach out to the maintainers.

---

**Thank you for contributing! 🙏**
