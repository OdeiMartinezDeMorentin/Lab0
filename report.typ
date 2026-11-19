#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.2cm, right: 2.2cm),
  header: context {
    if here().page() > 1 {
      text(9pt, fill: luma(120))[
        Lab 0: Fundamentals of Continuous Integration #h(1fr) Odei Martínez de Morentin
      ]
    }
  },
  footer: context {
    text(9pt, fill: luma(120))[
      #h(1fr) Page #counter(page).display()
    ]
  }
)

#set text(
  font: "Liberation Sans",
  size: 10pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.65em,
)

// Document Title & Metadata
#align(center)[
  #v(0.2cm)
  #text(18pt, weight: "bold")[Lab 0: Fundamentals of Continuous Integration] \
  #v(0.15cm)
  #text(11.5pt, fill: rgb("#334155"))[MLOps — Virtual Environments, Testing, Linting & Formatting] \
  #v(0.25cm)
  #line(length: 100%, stroke: 0.7pt + rgb("#cbd5e1"))
  #v(0.15cm)
  #grid(
    columns: (1fr, 1fr),
    align: (left, right),
    [ *Author:* Odei Martínez de Morentin \
      *Email:* #link("mailto:odeimardemor@gmail.com")[odeimardemor\@gmail.com] ],
    [ *Date:* November 2026 \
      *Repository:* #link("https://github.com/OdeiMartinezDeMorentin/Lab0.git")[Lab0 GitHub Repository] ]
  )
  #v(0.15cm)
  #line(length: 100%, stroke: 0.7pt + rgb("#cbd5e1"))
  #v(0.3cm)
]

= 1. Introduction and Objectives

In this first laboratory assignment, the primary objective was to get hands-on experience with the core pillars of *Continuous Integration (CI)* and modern MLOps practices. Instead of jumping straight into complex machine learning models, this lab focuses on building a solid software engineering foundation:

- Isolating project dependencies inside a dedicated virtual environment.
- Implementing a modular Python package with data pre-processing functions.
- Building a Command Line Interface (CLI) to interact with the logic easily.
- Ensuring code correctness through automated unit and integration tests.
- Enforcing formatting standards and static code quality rules before committing.

The complete code has been published to my public GitHub repository:
#align(center)[
  #link("https://github.com/OdeiMartinezDeMorentin/Lab0.git")
]

= 2. Project Setup and Decisions

== Dependency Management with `uv`
I chose to use *uv* for this project as required. Coming from standard `pip` and `venv`, `uv` makes project initialization and dependency synchronization much simpler and noticeably faster.

I initialized the project and added the required packages:
```bash
uv init
uv add click black pylint pytest pytest-cov
uv sync
```

Each package plays a specific role in our workflow:
- *Click:* For structuring our command line commands, options, and help messages.
- *Black:* An uncompromising code formatter that eliminates style debates by formatting code to PEP 8 automatically.
- *Pylint:* A static analyzer that checks for logic issues, unused variables, and style inconsistencies.
- *Pytest & Pytest-cov:* The testing framework and coverage plugin to verify our code and measure how much of it is actually tested.

== Project Structure
Following the scaffold provided in the lab guidelines, I organized the repository as follows:

```text
Lab0/
├── pyproject.toml        # Project metadata and dependencies definitions
├── pytest.ini            # Pytest configuration (pythonpath, testpaths, coverage options)
├── README.md             # Project documentation and CLI usage guide
├── .gitignore            # Ignored files (venv, caches, coverage artifacts)
├── uv.lock               # Pinned dependencies for reproducible installs
├── src/                  # Application source code
│   ├── __init__.py       # Package marker
│   ├── cli.py            # Command Line Interface (Click groups and commands)
│   └── preprocessing.py  # Core data preprocessing functions
└── tests/                # Automated tests
    ├── test_cli.py       # Integration tests for the CLI commands
    └── test_logic.py     # Unit tests for preprocessing functions
```

== Implementation of the Core Logic (`preprocessing.py`)
In `src/preprocessing.py`, I implemented the 13 required data pre-processing techniques, grouping them into:
1. *Data Cleaning:* Removing missing values (`remove_missing_values`), filling missing values with defaults or custom replacements (`fill_missing_values`), and extracting unique values (`remove_duplicates`).
2. *Numerical Scaling:* Min-max normalization (`min_max_normalize`), z-score standardization (`z_score_standardize`), value clipping (`clip_values`), converting string items to integers (`convert_to_integers`), and natural log transformation (`log_transform`).
3. *Text Processing:* Lowercase alphanumeric tokenization (`tokenize_text`), selecting alphanumeric characters and spaces (`select_alphanumeric_and_spaces` / `remove_punctuation`), and stop-word filtering (`remove_stop_words`).
4. *Data Structure Operations:* Flattening arbitrarily nested lists (`flatten_list`) and reproducible random shuffling with an optional seed (`shuffle_list`).

== Implementation of the CLI (`cli.py`)
To make the functionalities accessible from the terminal, I built a Click CLI in `src/cli.py`. I created a top-level `cli` group and four sub-groups:
- `clean`: houses `remove-missing` and `fill-missing` (with `--fill-value`, default `0`).
- `numeric`: houses `normalize`, `standardize`, `clip` (with `--min-val 0` and `--max-val 1`), `to-int`, and `log`.
- `text`: houses `tokenize`, `remove-punctuation`, and `remove-stopwords` (with `--stop-word`).
- `struct`: houses `shuffle` (with `--seed`), `flatten`, and `unique-values`.

Each command includes a clear description and a concrete usage example in its `--help` documentation.

= 3. Testing Strategy and Reasoning

The testing suite was designed using *pytest*. Rather than just checking that functions work with simple inputs (the "happy path"), I wanted to make sure they handle real-world messy data gracefully.

== 3.1 Why and How I Used Pytest Fixtures
Creating fake or repetitive data inside every single test makes tests cluttered and hard to maintain. Pytest *fixtures* solve this by providing reusable components that can be injected into test functions.

In `tests/test_logic.py`, I defined two data fixtures:
- `sample_numbers`: Provides a clean list of floats `[10.0, 20.0, 30.0]`. I used this in both `test_z_score_standardize_with_fixture` and `test_min_max_normalize_with_fixture`. It ensures that tests use predictable baseline numbers where manual calculations (mean, standard deviation, and range) are straightforward to verify.
- `sample_mixed_missing`: Provides a list with mixed valid values, zeroes, booleans, and missing values (`[1, None, "", float("nan"), 0, False, "hello", 42]`). This is used to test that filling functions only replace true missing values, without accidentally overwriting valid falsy values like `0` or `False`.

In `tests/test_cli.py`, I created the `runner` fixture:
- It returns an instance of Click's `CliRunner`. Instead of instantiating `runner = CliRunner()` at the beginning of every test function, having it as a fixture makes tests cleaner and ensures a fresh runner environment for each test.

== 3.2 Why and How I Used Parametrized Tests (`@pytest.mark.parametrize`)
Instead of duplicating test code to try different inputs, `@pytest.mark.parametrize` allows running the exact same test assertion across a table of test cases.

I used parametrization in both categories required by the assignment:

1. *Functionality without required options:*
   - `remove_missing_values`: I tested it with lists containing mixed nuls, lists containing *only* nuls, lists with valid data, and an empty list.
   - `tokenize_text`: I tested it with normal sentences, text containing numbers and punctuation, strings with only spaces, and empty strings.

2. *Functionality with options:*
   - `clip_values`: I tested clipping values below the minimum, above the maximum, values within range, and an empty list, using custom `min_val` and `max_val` boundaries.
   - `min_max_normalize`: I tested standard scaling to `[0.0, 1.0]`, custom scaling to `[-1.0, 1.0]`, and constant lists where minimum equals maximum.

== 3.3 Handling Edge Cases and Defensive Programming
Writing tests forced me to think about edge cases in the logic:
- *Division by zero in standard deviation:* When all numbers in a list are identical (e.g., `[7.0, 7.0, 7.0]`), the standard deviation is `0.0`. Dividing by zero would crash a pipeline, so I designed `z_score_standardize` to return zeros in this scenario and wrote a specific unit test (`test_z_score_standardize_constant_and_empty`) to verify it.
- *Invalid configuration ranges:* If a user calls `min_max_normalize` with `new_min > new_max` or `clip_values` with `min_val > max_val`, the logic raises a `ValueError`. In the tests, I used `pytest.raises(ValueError)` to verify that these safeguards trigger properly.
- *Logarithm of zero and negative numbers:* Natural logarithm is undefined for values $<= 0$. In `log_transform`, I filtered out non-positive and non-finite values so the function does not crash.
- *Falsy values vs missing values:* In Python, `0` and `False` are falsy, but they are valid data, not missing values. The unit tests verify that `remove_missing_values` preserves `0` and `False` while dropping `None`, `""`, and `NaN`.

== 3.4 Integration Testing of the CLI
Unit tests verify that Python functions work when called internally, but they do not guarantee that the Command Line Interface works for an end user.

Using `CliRunner`, I wrote integration tests covering all four CLI groups (`clean`, `numeric`, `text`, and `struct`). These tests verify:
- That the command returns an exit code of `0` (success).
- That arguments and options (e.g., `--new-min`, `--new-max`, `--seed`, `--fill-value`) are parsed correctly by Click.
- That the CLI output printed to the terminal matches what the user expects.

= 4. Linting, Formatting, and Testing Results

To verify the quality and reliability of the code, I ran the three required verification tools from the assignment instructions. Below are the execution commands and their outputs.

== 4.1 Automatic Code Formatting (Black)
Black reformats code to follow PEP 8 rules strictly (consistent line length, quotes, and indentation).

*Command executed:*
```bash
uv run black --check src/*.py
```

*Output:*
```text
All done! ✨ 🍰 ✨
3 files would be left unchanged.
```
All source files in `src/` (`__init__.py`, `cli.py`, `preprocessing.py`) conform to the Black style standard without requiring any changes.

== 4.2 Static Code Analysis and Linting (Pylint)
Pylint inspects Python code for errors, stylistic issues, unused imports, and bad coding patterns.

*Command executed:*
```bash
uv run python -m pylint src/*.py
```

*Output:*
```text
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
The codebase scored a perfect *10.00 / 10.00*. All functions have clear docstrings, appropriate variable names, and explicit type hints.

== 4.3 Automated Testing and Coverage (Pytest & Pytest-cov)
Pytest executes all test cases, while Pytest-cov measures the proportion of source code executed during testing.

*Command executed:*
```bash
uv run python -m pytest -v --cov=src
```

*Output summary:*
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- .venv/bin/python3
configfile: pytest.ini
testpaths: tests
plugins: cov-7.1.0, platformdirs-4.14.5
collected 34 items

tests/test_cli.py (5 tests: help, clean, numeric, text, struct) ......... PASSED
tests/test_logic.py (29 unit & parametrized tests) ...................... PASSED

================================ tests coverage ================================
Name                   Stmts   Miss  Cover   Missing
----------------------------------------------------
src/__init__.py            0      0   100%
src/cli.py                83      0   100%
src/preprocessing.py      94      0   100%
----------------------------------------------------
TOTAL                    177      0   100%
============================== 34 passed in 0.27s ==============================
```

All *34 tests passed*, achieving *100% test coverage* across all modules in `src/`.

= 5. Conclusion and Personal Reflections

This lab provided a clear view of why Continuous Integration tools are essential before starting any machine learning work:
1. *Reproducibility:* Using `uv` guarantees that anyone cloning the repository can reproduce the exact same environment and package versions.
2. *Confidence through Testing:* Having a test suite with 100% coverage means that if I need to optimize a pre-processing function later, I can refactor it with confidence that existing behavior will not break.
3. *Automated Standards:* Linters and formatters remove the mental burden of formatting code manually and catch subtle bugs before code is even committed.

Overall, the project is fully structured, tested, documented, and ready for deployment into automated CI pipelines in future labs.
