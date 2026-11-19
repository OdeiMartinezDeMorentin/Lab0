"""Unit tests for the data preprocessing module."""

# pylint: disable=redefined-outer-name

import math
from typing import Any, List

import pytest

from src.preprocessing import (
    clip_values,
    convert_to_integers,
    fill_missing_values,
    flatten_list,
    log_transform,
    min_max_normalize,
    remove_duplicates,
    remove_missing_values,
    remove_punctuation,
    remove_stop_words,
    select_alphanumeric_and_spaces,
    shuffle_list,
    tokenize_text,
    z_score_standardize,
)


# Fixtures
@pytest.fixture
def sample_numbers() -> List[float]:
    """Generate a standard list of numbers for numerical preprocessing tests."""
    return [10.0, 20.0, 30.0]


@pytest.fixture
def sample_mixed_missing() -> List[Any]:
    """Generate a list containing mixed valid and missing values."""
    return [1, None, "", float("nan"), 0, False, "hello", 42]


# Parametrized tests
@pytest.mark.parametrize(
    "input_values, expected",
    [
        ([1, None, 2, "", 3], [1, 2, 3]),
        ([None, "", float("nan")], []),
        ([0, False, "valid"], [0, False, "valid"]),
        ([], []),
    ],
)
def test_remove_missing_values_parametrized(
    input_values: List[Any], expected: List[Any]
) -> None:
    """Parametrized unit tests for remove_missing_values."""
    assert remove_missing_values(input_values) == expected


@pytest.mark.parametrize(
    "raw_text, expected_tokens",
    [
        ("Hello, World!", ["hello", "world"]),
        ("MLOps 2026: Lab_0!", ["mlops", "2026", "lab", "0"]),
        ("   ", []),
        ("", []),
        ("Tokenize123 and-more", ["tokenize123", "and", "more"]),
    ],
)
def test_tokenize_text_parametrized(raw_text: str, expected_tokens: List[str]) -> None:
    """Parametrized unit tests for tokenize_text."""
    assert tokenize_text(raw_text) == expected_tokens


@pytest.mark.parametrize(
    "values, min_val, max_val, expected",
    [
        ([-5.0, 0.0, 5.0, 15.0], 0.0, 10.0, [0.0, 0.0, 5.0, 10.0]),
        ([1.0, 2.0, 3.0], 1.0, 3.0, [1.0, 2.0, 3.0]),
        ([10.0, 20.0], 12.0, 15.0, [12.0, 15.0]),
        ([], 0.0, 1.0, []),
    ],
)
def test_clip_values_parametrized(
    values: List[float], min_val: float, max_val: float, expected: List[float]
) -> None:
    """Parametrized unit tests for clip_values (function with required options)."""
    assert clip_values(values, min_val=min_val, max_val=max_val) == expected


@pytest.mark.parametrize(
    "values, new_min, new_max, expected",
    [
        ([0.0, 5.0, 10.0], 0.0, 1.0, [0.0, 0.5, 1.0]),
        ([10.0, 20.0, 30.0], -1.0, 1.0, [-1.0, 0.0, 1.0]),
        ([5.0, 5.0], 0.0, 1.0, [0.0, 0.0]),
    ],
)
def test_min_max_normalize_parametrized(
    values: List[float], new_min: float, new_max: float, expected: List[float]
) -> None:
    """Parametrized unit tests for min_max_normalize (function with options)."""
    assert min_max_normalize(values, new_min=new_min, new_max=new_max) == pytest.approx(
        expected
    )


# Tests with fixtures
def test_z_score_standardize_with_fixture(sample_numbers: List[float]) -> None:
    """Test z-score standardization using sample_numbers fixture."""
    standardized = z_score_standardize(sample_numbers)
    expected_std = math.sqrt(200.0 / 3.0)
    assert standardized == pytest.approx(
        [-10.0 / expected_std, 0.0, 10.0 / expected_std]
    )


def test_min_max_normalize_with_fixture(sample_numbers: List[float]) -> None:
    """Test min-max normalization using sample_numbers fixture."""
    normalized = min_max_normalize(sample_numbers, new_min=0.0, new_max=1.0)
    assert normalized == pytest.approx([0.0, 0.5, 1.0])


def test_fill_missing_values_with_fixture(sample_mixed_missing: List[Any]) -> None:
    """Test filling missing values using sample_mixed_missing fixture."""
    filled_default = fill_missing_values(sample_mixed_missing)
    assert filled_default == [1, 0, 0, 0, 0, False, "hello", 42]

    filled_custom = fill_missing_values(sample_mixed_missing, fill_value="MISSING")
    assert filled_custom == [
        1,
        "MISSING",
        "MISSING",
        "MISSING",
        0,
        False,
        "hello",
        42,
    ]


def test_remove_duplicates() -> None:
    """Test removing duplicates while preserving order for hashable and unhashable items."""
    assert remove_duplicates([3, 1, 2, 1, 3, 4]) == [3, 1, 2, 4]
    assert remove_duplicates(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert remove_duplicates([[1, 2], [3], [1, 2]]) == [[1, 2], [3]]
    assert not remove_duplicates([])


def test_min_max_normalize_invalid_range() -> None:
    """Test min_max_normalize raises ValueError when new_min > new_max."""
    with pytest.raises(
        ValueError, match="new_min must be less than or equal to new_max"
    ):
        min_max_normalize([1.0, 2.0], new_min=5.0, new_max=1.0)
    assert min_max_normalize([]) == []


def test_z_score_standardize_constant_and_empty() -> None:
    """Test z-score standardization with constant values and empty list."""
    assert z_score_standardize([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]
    assert z_score_standardize([]) == []


def test_clip_values_invalid_range() -> None:
    """Test clip_values raises ValueError when min_val > max_val."""
    with pytest.raises(
        ValueError, match="min_val must be less than or equal to max_val"
    ):
        clip_values([1.0, 2.0], min_val=5.0, max_val=2.0)


def test_convert_to_integers() -> None:
    """Test converting numerical strings to integers and excluding non-numerical strings."""
    data = ["10", "-5", "3.9", "abc", "", "nan", "inf", " 42 "]
    assert convert_to_integers(data) == [10, -5, 3, 42]
    assert not convert_to_integers([])


def test_log_transform() -> None:
    """Test logarithmic transformation keeping only strictly positive finite numbers."""
    data = [-2, 0, 1, math.e, math.e**2, float("nan"), float("inf")]
    assert log_transform(data) == pytest.approx([0.0, 1.0, 2.0])
    assert log_transform([-1, 0]) == []


def test_select_alphanumeric_and_spaces() -> None:
    """Test selecting only alphanumeric characters and spaces."""
    raw = "Hello, World! 123 #MLOps_2026?"
    assert select_alphanumeric_and_spaces(raw) == "Hello World 123 MLOps2026"
    assert remove_punctuation(raw) == "Hello World 123 MLOps2026"
    assert select_alphanumeric_and_spaces("") == ""


def test_remove_stop_words() -> None:
    """Test removing stop words from lowercased text."""
    raw = "The Quick Brown Fox jumps over the lazy dog"
    result = remove_stop_words(raw, ["the", "OVER"])
    assert result == "quick brown fox jumps lazy dog"
    assert remove_stop_words("", ["the"]) == ""


def test_flatten_list() -> None:
    """Test flattening a list of lists (including nested and empty sublists)."""
    assert flatten_list([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
    assert flatten_list([[1, [2, 3]], [], [4]]) == [1, 2, 3, 4]
    assert flatten_list([["a", "b"], ["c"]]) == ["a", "b", "c"]
    assert not flatten_list([])


def test_shuffle_list_reproducibility() -> None:
    """Test reproducible shuffling with seed without mutating input list."""
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    shuffled_a = shuffle_list(original, seed=42)
    shuffled_b = shuffle_list(original, seed=42)

    assert shuffled_a == shuffled_b
    assert sorted(shuffled_a) == original
    assert original == [1, 2, 3, 4, 5, 6, 7, 8]
    assert not shuffle_list([], seed=42)
