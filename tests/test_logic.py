"""Unit tests for the data preprocessing module."""

import math

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
    remove_stop_words,
    select_alphanumeric_and_spaces,
    shuffle_list,
    tokenize_text,
    z_score_standardize,
)


def test_remove_missing_values() -> None:
    """Test removing None, empty string, and NaN while preserving 0 and False."""
    data = [1, None, "", float("nan"), 0, False, "valid", 3.5]
    assert remove_missing_values(data) == [1, 0, False, "valid", 3.5]
    assert remove_missing_values([]) == []


def test_fill_missing_values_default_and_custom() -> None:
    """Test filling missing values with default (0) and custom values."""
    data = [1, None, "", float("nan"), 5]
    assert fill_missing_values(data) == [1, 0, 0, 0, 5]
    assert fill_missing_values(data, fill_value=-1) == [1, -1, -1, -1, 5]
    assert fill_missing_values([0, False], fill_value=99) == [0, False]


def test_remove_duplicates() -> None:
    """Test removing duplicates while preserving order for hashable and unhashable items."""
    assert remove_duplicates([3, 1, 2, 1, 3, 4]) == [3, 1, 2, 4]
    assert remove_duplicates(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert remove_duplicates([[1, 2], [3], [1, 2]]) == [[1, 2], [3]]
    assert not remove_duplicates([])


def test_min_max_normalize_default_and_custom() -> None:
    """Test min-max normalization with default [0.0, 1.0] and custom ranges."""
    assert min_max_normalize([0.0, 5.0, 10.0]) == pytest.approx([0.0, 0.5, 1.0])
    assert min_max_normalize([10.0, 20.0, 30.0], new_min=-1.0, new_max=1.0) == (
        pytest.approx([-1.0, 0.0, 1.0])
    )


def test_min_max_normalize_edge_cases() -> None:
    """Test min-max normalization with empty, constant, and invalid range inputs."""
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0, 5.0, 5.0], new_min=2.0, new_max=4.0) == [
        2.0,
        2.0,
        2.0,
    ]
    with pytest.raises(ValueError):
        min_max_normalize([1.0, 2.0], new_min=5.0, new_max=1.0)


def test_z_score_standardize() -> None:
    """Test z-score standardization for normal, constant, and empty lists."""
    standardized = z_score_standardize([10.0, 20.0, 30.0])
    expected_std = math.sqrt(200.0 / 3.0)
    assert standardized == pytest.approx(
        [-10.0 / expected_std, 0.0, 10.0 / expected_std]
    )
    assert z_score_standardize([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]
    assert z_score_standardize([]) == []


def test_clip_values() -> None:
    """Test clipping numerical values and invalid range handling."""
    assert clip_values([-5.0, 0.0, 3.0, 10.0, 15.0], min_val=0.0, max_val=10.0) == [
        0.0,
        0.0,
        3.0,
        10.0,
        10.0,
    ]
    assert clip_values([], min_val=0.0, max_val=1.0) == []
    with pytest.raises(ValueError):
        clip_values([1.0, 2.0], min_val=5.0, max_val=2.0)


def test_convert_to_integers() -> None:
    """Test converting numerical strings to integers and excluding non-numerical strings."""
    data = ["10", "-5", "3.9", "abc", "", "nan", "inf", " 42 "]
    assert convert_to_integers(data) == [10, -5, 3, 42]
    assert not convert_to_integers([])


def test_log_transform() -> None:
    """Test logarithmic transformation keeping only positive finite numbers."""
    data = [-2, 0, 1, math.e, math.e**2, float("nan"), float("inf")]
    assert log_transform(data) == pytest.approx([0.0, 1.0, 2.0])
    assert log_transform([-1, 0]) == []


def test_tokenize_text() -> None:
    """Test tokenization into lowercased alphanumeric words."""
    raw = "Hello, World! 2026 MLOps_Lab."
    assert tokenize_text(raw) == ["hello", "world", "2026", "mlops", "lab"]
    assert tokenize_text("") == []
    assert tokenize_text("... !!! ___") == []


def test_select_alphanumeric_and_spaces() -> None:
    """Test selecting only alphanumeric characters and spaces."""
    raw = "Hello, World! 123 #MLOps_2026?"
    assert select_alphanumeric_and_spaces(raw) == "Hello World 123 MLOps2026"
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


def test_shuffle_list_reproducibility_and_immutability() -> None:
    """Test reproducible shuffling with seed without mutating input list."""
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    shuffled_a = shuffle_list(original, seed=42)
    shuffled_b = shuffle_list(original, seed=42)

    assert shuffled_a == shuffled_b
    assert sorted(shuffled_a) == original
    assert original == [1, 2, 3, 4, 5, 6, 7, 8]
    assert not shuffle_list([], seed=42)
