# Contributing to Forge

Thank you for your interest in contributing to Forge! This guide will help you get started.

## Development Setup

### Prerequisites

- **Python 3.10–3.12** (3.13+ is not supported)
- **Rust 1.75+** (for building `forge-core`)
- **Git** (2.x+)

### Clone & Install

```bash
git clone https://github.com/Paramveersingh-S/Forge.git
cd Forge

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Build the Rust core
cargo build --manifest-path forge-core/Cargo.toml

# Install pre-commit hooks
pre-commit install
```

## Workflow

### 1. Pick an Issue

- Look for issues tagged `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it
- If there's no issue for your change, create one first

### 2. Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/your-fix-name
```

### 3. Write Code

- Follow the existing code style (enforced by `ruff`)
- Add type hints to all new functions
- Write docstrings for public APIs
- Keep modules focused — one concern per file

### 4. Write Tests

Every change must include tests. We use three test tiers:

| Marker | Speed | Requirements | Command |
|:---|:---|:---|:---|
| `unit` | Fast (~30s total) | No GPU, no network | `pytest -m unit` |
| `integration` | Medium (~2min) | May use filesystem/subprocess | `pytest -m integration` |
| `smoke` | Slow (~10min) | Requires GPU + model download | `pytest -m smoke --gpu` |

**Minimum**: Unit tests for all new logic. Integration tests for CLI commands.

```bash
# Run all fast tests
pytest tests/ -m "not smoke and not gpu"

# Run with coverage
pytest tests/ --cov=forge --cov-report=html

# Run Rust tests
cargo test --manifest-path forge-core/Cargo.toml
```

### 5. Lint & Format

```bash
# Python
ruff check src/ tests/
ruff format src/ tests/

# Rust
cargo clippy --manifest-path forge-core/Cargo.toml -- -D warnings
cargo fmt --manifest-path forge-core/Cargo.toml --check

# Type checking
mypy src/forge/
```

### 6. Submit a PR

- PR title should follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat: add KTO trainer`
  - `fix: correct VRAM estimation for NF4 streaming`
  - `docs: add layer streaming tutorial`
  - `test: add unit tests for config fragment merging`
- Fill out the PR template
- All CI checks must pass before merge

## Architecture Principles

1. **Separation of concerns**: Config parsing ≠ training logic ≠ I/O operations
2. **No top-level torch imports in CLI path**: The light CLI (`forge --help`, `forge doctor`) must not import PyTorch
3. **Rust for I/O, Python for orchestration**: The Rust core handles memory-mapped I/O, DMA scheduling, and crypto. Python handles training loop orchestration and HuggingFace integration.
4. **Composability over monolithism**: Prefer small, focused modules over large multi-concern files
5. **Test the artifact, not just the code**: If a training run produces an adapter, test that the adapter loads and generates, not just that the loss decreased

## Code Review Checklist

- [ ] Tests pass locally
- [ ] New code has type hints
- [ ] Public functions have docstrings
- [ ] No top-level torch/transformers imports in `cli/` or `config/` modules
- [ ] Config schema changes include migration notes
- [ ] Breaking changes are documented in the PR description

## Getting Help

- Open a [Discussion](https://github.com/Paramveersingh-S/Forge/discussions) for questions
- Tag `@Paramveersingh-S` for architecture questions
- Check existing issues and PRs before opening a new one

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
