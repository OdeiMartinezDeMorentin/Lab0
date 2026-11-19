# Lab 0: MLOps Project Environment & Best Practices

This project implements the foundational environment setup and best practices for MLOps, including dependency management via `uv`, a modular source structure with a Command Line Interface (CLI), comprehensive testing, code linting, formatting, and test coverage tracking.

---

## Project Structure

The project follows the scaffold specified for MLOps best practices:

```text
Lab0/
├── pyproject.toml        # Project configuration and dependency definitions
├── pytest.ini            # Pytest configuration (pythonpath, testpaths, coverage options)
├── README.md             # Project documentation and usage guide
├── src/                  # Source code for business logic and CLI
│   ├── __init__.py       # Package initializer
│   ├── cli.py            # Command Line Interface developed with Click
│   └── preprocessing.py  # Data preprocessing and logic functions
├── tests/                # Test suite
│   ├── test_cli.py       # Integration tests for the CLI
│   └── test_logic.py     # Unit tests for preprocessing logic
├── uv.lock               # Deterministic dependency lockfile
└── .venv/                # Virtual environment managed by uv
```

---

## Dependencies

The project relies on specific packages installed via `uv add`:

| Package | Category | Purpose |
| :--- | :--- | :--- |
| **Click** | CLI Framework | Develops the command line interface to interact with the project logic. |
| **Black** | Code Formatter | Enforces deterministic PEP 8 code formatting across the repository. |
| **Pylint** | Linter | Analyzes code for errors, stylistic issues, and adherence to coding standards. |
| **Pytest** | Test Framework | Executes both unit and integration tests. |
| **Pytest-cov** | Coverage Plugin | Measures and reports test code coverage across the source files. |

---

## Environment Setup

### 1. Initialize and Sync the Virtual Environment

Initialize the project and synchronize the virtual environment:

```bash
# Initialize uv project (already completed)
uv init

# Install project dependencies into the virtual environment
uv sync
```

### 2. Dependency Management

To add new dependencies or tools:

```bash
uv add <library-name>
```

---

## CLI Usage

The CLI is built with `Click` and provides subcommands to interact with preprocessing routines.

Run the CLI using `uv run`:

```bash
# Display help and available command groups
uv run python -m src.cli --help

# Cleaning commands
uv run python -m src.cli clean remove-missing 1 None "" nan 2
uv run python -m src.cli clean fill-missing 1 None "" nan --fill-value 0
uv run python -m src.cli clean remove-duplicates a b a c

# Numerical commands
uv run python -m src.cli numeric normalize 0 5 10 --new-min 0.0 --new-max 1.0
uv run python -m src.cli numeric standardize 10 20 30
uv run python -m src.cli numeric clip --min-val 0 --max-val 10 -- -5 3 15
uv run python -m src.cli numeric to-int 10 abc 20 -5
uv run python -m src.cli numeric log -- -1 0 1 2.71828

# Text commands
uv run python -m src.cli text tokenize "Hello, World! 2026"
uv run python -m src.cli text alphanumeric "Hello, #MLOps_2026!"
uv run python -m src.cli text remove-stopwords "The quick brown Fox" --stop-word the

# List structure commands
uv run python -m src.cli struct flatten "[[1, 2], [3, 4]]"
uv run python -m src.cli struct shuffle a b c d --seed 42
```

---

## Code Quality & Testing

### Code Formatting (Black)

Check code formatting:
```bash
uv run black --check .
```

Automatically reformat code:
```bash
uv run black .
```

### Code Linting (Pylint)

Run Pylint on source and test files:
```bash
uv run pylint src tests
```

### Automated Testing & Coverage (Pytest & Pytest-cov)

Run all unit and integration tests with coverage reporting:
```bash
uv run pytest
```
