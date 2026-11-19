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

The project testing was divided into two parts. First, unit testing was used for the pre-processing functions. Second, integration testing was used for the command line interface. The goal was to check that all functions work with normal data and with bad or missing data.

In "test_logic.py", tests were written for each function. Fixtures were used to share test data between different tests without writing the same list multiple times. For example, the sample_numbers fixture gives a list of three numbers (10.0, 20.0, 30.0). This list was reused in the tests for z-score standardization and min-max normalization. The sample_mixed_missing fixture gives a list with valid data, empty strings, None, and NaN. It also includes 0 and False. This checks that cleaning functions only remove actual missing values and do not delete valid numbers or booleans.

The parametrize decorator from pytest was used to test functions with several inputs in a single test. This avoids writing repeated test functions.

For functions without options, like remove_missing_values, tests checked lists with mixed missing values, lists with only missing values, normal lists, and empty lists. For tokenize_text, tests checked text with punctuation, capital letters, extra spaces, and empty text.

For functions with options, like clip_values, parametrization tested values below the minimum, above the maximum, and inside the range. For min_max_normalize, tests checked the default range from 0 to 1, a custom range from -1 to 1, and lists where all numbers are equal.

Edge cases were also tested. In z_score_standardize, when all numbers are the same, the standard deviation is zero. The function returns zeros instead of crashing with a division by zero error. In min_max_normalize and clip_values, if the minimum is bigger than the maximum, the function raises a ValueError. This error was verified with pytest.raises.

The second part of the testing is in "test_cli.py". Unit tests only check Python functions, but they do not check if the terminal commands work. The CLI can fail if inputs or options are not read correctly from the terminal. A fixture for CliRunner was created to run commands inside the tests. This runner was used to test all four command groups: clean, numeric, text, and struct. The tests verify that each command finishes with exit code 0, reads options like fill-value, new-min, new-max, and seed correctly, and prints the expected result to the terminal.
