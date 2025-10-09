# Contributing to Reminiscence

Thank you for considering contributing to Reminiscence. This guide covers the development workflow and technical requirements.

For usage documentation and API reference, visit [reminiscence.dev](https://reminiscence.dev).

## Development Setup
### Recommended: Dev Container

The easiest way to get started is using our pre-configured dev container with all tools and dependencies installed:

```bash
# Using the Demiotic dev container
docker pull ghcr.io/demiotic/dev-reminiscence:latest

# Or let your IDE handle it (VS Code, etc.)
# The .devcontainer configuration will automatically use the image
```
The dev container includes Python 3.12, uv, ruff, mypy, pytest, and Docker-in-Docker for OTEL tests.

### Manual Setup
If you prefer local development:

#### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Docker (optional, for OpenTelemetry integration tests)

#### Installation
```
git clone https://github.com/demiotic/reminiscence.git
cd reminiscence

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up pre-commit hooks (recommended)
uv pip install pre-commit
pre-commit install
```

## Running Tests

### Standard Test Suite

```bash
# Run all tests (~4 minutes, 257 tests)
uv run pytest

# With coverage report
uv run pytest --cov=reminiscence --cov-report=html

# Parallel execution
uv run pytest -n auto
```

### OpenTelemetry Integration Tests

Tests requiring OpenTelemetry automatically start a Docker collector. If you see connection errors after tests complete, this is expected behavior from background threads attempting to export metrics after Docker shutdown—the tests themselves have passed.

Manual collector setup for debugging:

```bash
docker run -d --rm --name otel-collector \
    -p 4318:4318 -p 4317:4317 \
    otel/opentelemetry-collector-contrib:latest

uv run pytest tests/test_metrics.py

docker stop otel-collector
```

## Code Standards

### Automated Checks
Pre-commit hooks automatically run before each commit. To run manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Format code
uv run ruff format .

# Lint with auto-fix
uv run ruff check . --fix

# Type checking
uv run mypy reminiscence
```

### Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add similarity_threshold override to lookup()
fix: prevent caching None results by default
docs: update batch embeddings example
test: add error handling coverage for store_batch()
perf: optimize batch embedding generation (3x improvement)
```

## Pull Request Process

Before submitting:

- [ ] Tests pass: `uv run pytest`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Code is formatted: `uv run ruff format .`
- [ ] Linting is clean: `uv run ruff check .`
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for user-facing changes

CI must pass before review. Respond to feedback constructively and make requested changes in additional commits.


## Testing Guidelines

- Use provided fixtures from `conftest.py`
- Keep tests isolated with proper cleanup
- Use descriptive test names: `test_store_skips_error_results_by_default`
- Follow Arrange-Act-Assert structure
- Cover edge cases and error conditions

Example test:

```python
def test_store_skips_errors_by_default(self, reminiscence):
"""store() should skip error results by default."""
    reminiscence.store("query", {}, {"error": "failed"})

    result = reminiscence.lookup("query", {})
    assert result.is_miss
    assert reminiscence.metrics.store_errors == 1
```

## Questions and Support

- Check [existing issues](https://github.com/demiotic/reminiscence/issues)
- Review [discussions](https://github.com/demiotic/reminiscence/discussions)
- Open a new issue with the `question` label
---

Thank you for contributing to Reminiscence.