#set page(
  paper: "a4",
  margin: 2cm,
  header: context {
    if here().page() > 1 {
      text(9pt, fill: luma(100))[
        Lab 0: Fundamentals of Continuous Integration #h(1fr) Odei Martínez de Morentin
      ]
    }
  },
  footer: context {
    text(9pt, fill: luma(100))[
      #h(1fr) #counter(page).display()
    ]
  }
)

#set text(
  font: "Liberation Sans",
  size: 11pt,
  lang: "en",
)

#show raw.where(block: true): it => {
  set text(size: 6.5pt)
  set par(leading: 0.4em)
  it
}

#set par(
  justify: true,
  leading: 0.75em,
)

#align(center)[
  #text(16pt, weight: "bold")[Lab 0: Fundamentals of Continuous Integration] \
  #v(0.1cm)
  #text(11pt)[Odei Martínez de Morentin] \
  #text(10pt)[GitHub Repository: https://github.com/OdeiMartinezDeMorentin/Lab0.git]
  #v(0.5cm)
]

= 1. Testing Logic and Strategy

The project testing is divided into two parts. First, unit testing verifies the core logic of the pre-processing functions. Second, integration testing checks that the command line interface interacts properly with that logic and handles user commands as expected. The goal is to check that all functions work with normal data and with bad or missing data.

In "test_logic.py", tests cover each function. Fixtures share test data between different tests without writing the same list multiple times. For example, the sample_numbers fixture gives a list of three numbers (10.0, 20.0, 30.0). This list is reused in the tests for z-score standardization and min-max normalization. The sample_mixed_missing fixture gives a list with valid data, empty strings, None, and NaN. It also includes 0 and False. This checks that cleaning functions only remove actual missing values and do not delete valid numbers or booleans.

The parametrize decorator from pytest tests functions with several inputs in a single test function, which avoids repeating test code. This is applied to two different types of functions. Functions without options, like remove_missing_values and tokenize_text, only take raw data as input, so the tests only vary the input data (such as nulls, punctuation, and empty lists). In contrast, functions with options, like clip_values and min_max_normalize, require extra configuration settings. These tests are different because they must vary both the input numbers and the option parameters at the same time, testing different clipping boundaries (min_val and max_val), custom normalization ranges (new_min and new_max), and invalid settings where the minimum is larger than the maximum.

Edge cases are also tested. In z_score_standardize, when all numbers are the same, the standard deviation is zero. The function returns zeros instead of crashing with a division by zero error. In min_max_normalize and clip_values, if the minimum is bigger than the maximum, the function raises a ValueError. This error is verified with pytest.raises.

The second part of the testing is in "test_cli.py". Unit tests only check Python functions, but they do not check if the terminal commands work. The CLI can fail if inputs or options are not read correctly from the terminal. A fixture for CliRunner runs commands inside the tests. This runner tests all four command groups: clean, numeric, text, and struct. The tests verify that each command finishes with exit code 0, reads options like fill-value, new-min, new-max, and seed correctly, and prints the expected result to the terminal.

#v(0.4cm)

= 2. Results of Formatting, Linting, and Testing

Three tools check the code: Black, Pylint, and Pytest with pytest-cov.

== 2.1 Code Formatting (Black)
Black checks that the code style follows PEP 8.

Execution:
```text
(lab0) alumno@eim-alu-83080:~/Desktop/datos/repos/MLOPS/Lab0$ uv run black --check src/*.py
All done! ✨ 🍰 ✨
3 files would be left unchanged.
```

== 2.2 Linting (Pylint)
Pylint checks for errors, unused variables, and style problems.

Execution:
```text
(lab0) alumno@eim-alu-83080:~/Desktop/datos/repos/MLOPS/Lab0$ uv run python -m pylint src/*.py

--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

== 2.3 Testing and Coverage (Pytest and Pytest-cov)
Pytest runs all tests, and pytest-cov measures code coverage.

Execution:
```text
(lab0) alumno@eim-alu-83080:~/Desktop/datos/repos/MLOPS/Lab0$ uv run python -m pytest -v --cov=src
============================================ test session starts =============================================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /home/alumno/Desktop/datos/repos/MLOPS/Lab0/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/alumno/Desktop/datos/repos/MLOPS/Lab0
configfile: pytest.ini
testpaths: tests
plugins: cov-7.1.0, platformdirs-4.14.5
collected 34 items                                                                                           

tests/test_cli.py::test_cli_help_and_version PASSED                                                    [  2%]
tests/test_cli.py::test_cli_clean_commands PASSED                                                      [  5%]
tests/test_cli.py::test_cli_numeric_commands PASSED                                                    [  8%]
tests/test_cli.py::test_cli_text_commands PASSED                                                       [ 11%]
tests/test_cli.py::test_cli_struct_commands PASSED                                                     [ 14%]
tests/test_logic.py::test_remove_missing_values_parametrized[input_values0-expected0] PASSED           [ 17%]
tests/test_logic.py::test_remove_missing_values_parametrized[input_values1-expected1] PASSED           [ 20%]
tests/test_logic.py::test_remove_missing_values_parametrized[input_values2-expected2] PASSED           [ 23%]
tests/test_logic.py::test_remove_missing_values_parametrized[input_values3-expected3] PASSED           [ 26%]
tests/test_logic.py::test_tokenize_text_parametrized[Hello, World!-expected_tokens0] PASSED            [ 29%]
tests/test_logic.py::test_tokenize_text_parametrized[MLOps 2026: Lab_0!-expected_tokens1] PASSED       [ 32%]
tests/test_logic.py::test_tokenize_text_parametrized[   -expected_tokens2] PASSED                      [ 35%]
tests/test_logic.py::test_tokenize_text_parametrized[-expected_tokens3] PASSED                         [ 38%]
tests/test_logic.py::test_tokenize_text_parametrized[Tokenize123 and-more-expected_tokens4] PASSED     [ 41%]
tests/test_logic.py::test_clip_values_parametrized[values0-0.0-10.0-expected0] PASSED                  [ 44%]
tests/test_logic.py::test_clip_values_parametrized[values1-1.0-3.0-expected1] PASSED                   [ 47%]
tests/test_logic.py::test_clip_values_parametrized[values2-12.0-15.0-expected2] PASSED                 [ 50%]
tests/test_logic.py::test_clip_values_parametrized[values3-0.0-1.0-expected3] PASSED                   [ 52%]
tests/test_logic.py::test_min_max_normalize_parametrized[values0-0.0-1.0-expected0] PASSED             [ 55%]
tests/test_logic.py::test_min_max_normalize_parametrized[values1--1.0-1.0-expected1] PASSED            [ 58%]
tests/test_logic.py::test_min_max_normalize_parametrized[values2-0.0-1.0-expected2] PASSED             [ 61%]
tests/test_logic.py::test_z_score_standardize_with_fixture PASSED                                      [ 64%]
tests/test_logic.py::test_min_max_normalize_with_fixture PASSED                                        [ 67%]
tests/test_logic.py::test_fill_missing_values_with_fixture PASSED                                      [ 70%]
tests/test_logic.py::test_remove_duplicates PASSED                                                     [ 73%]
tests/test_logic.py::test_min_max_normalize_invalid_range PASSED                                       [ 76%]
tests/test_logic.py::test_z_score_standardize_constant_and_empty PASSED                                [ 79%]
tests/test_logic.py::test_clip_values_invalid_range PASSED                                             [ 82%]
tests/test_logic.py::test_convert_to_integers PASSED                                                   [ 85%]
tests/test_logic.py::test_log_transform PASSED                                                         [ 88%]
tests/test_logic.py::test_select_alphanumeric_and_spaces PASSED                                        [ 91%]
tests/test_logic.py::test_remove_stop_words PASSED                                                     [ 94%]
tests/test_logic.py::test_flatten_list PASSED                                                          [ 97%]
tests/test_logic.py::test_shuffle_list_reproducibility PASSED                                          [100%]

=============================================== tests coverage ===============================================
______________________________ coverage: platform linux, python 3.12.3-final-0 _______________________________

Name                   Stmts   Miss  Cover   Missing
----------------------------------------------------
src/__init__.py            0      0   100%
src/cli.py                83      0   100%
src/preprocessing.py      94      0   100%
----------------------------------------------------
TOTAL                    177      0   100%
============================================= 34 passed in 0.30s =============================================
```

In summary, the results confirm complete adherence to continuous integration standards. Black confirms that all source files follow PEP 8 formatting without any changes. Pylint scores 10.00/10, which proves the absence of syntax errors, unused variables, and style issues. Finally, Pytest executes all 34 tests successfully, and pytest-cov reports 100% code coverage across "preprocessing.py", "cli.py", and "\_\_init\_\_.py". This guarantees that both the core logic and the command line interface behave correctly and handle edge cases safely.
