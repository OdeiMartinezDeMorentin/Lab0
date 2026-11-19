# Lab 0: Fundamentals of Continuous Integration

Repository for Lab 0 of the MLOps course, implementing environment management with `uv`, modular preprocessing logic, a `click` CLI, code formatting (`black`), linting (`pylint`), and automated testing (`pytest`, `pytest-cov`).

## Setup

Synchronize the virtual environment and install dependencies:

```bash
uv sync
```

## CLI Usage

The command line interface exposes data preprocessing routines organized into four groups (`clean`, `numeric`, `text`, `struct`):

```bash
# Main help
uv run python -m src.cli --help

# Group help (e.g., clean)
uv run python -m src.cli clean --help
```

## Code Quality & Testing

Run code formatting checks:
```bash
uv run black --check src/*.py
```

Run linting:
```bash
uv run python -m pylint src/*.py
```

Run tests with code coverage:
```bash
uv run python -m pytest -v --cov=src
```
