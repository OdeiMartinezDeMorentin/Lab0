"""Data preprocessing module for the MLOps project.

Provides preprocessing techniques for missing data, deduplication,
numerical scaling/transformations, text processing, and list operations.
"""

import math
import random
import re
from typing import Any, Iterable, List, Optional


def _is_missing(value: Any) -> bool:
    """Return True if value represents a missing value (None, '', or NaN).

    Args:
        value: Value to inspect.

    Returns:
        True if the value is missing, False otherwise.
    """
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def remove_missing_values(values: List[Any]) -> List[Any]:
    """Remove missing values (None, '', and NaN) from a list.

    Args:
        values: Input list potentially containing missing values.

    Returns:
        List of values with missing entries removed.
    """
    return [value for value in values if not _is_missing(value)]


def fill_missing_values(values: List[Any], fill_value: Any = 0) -> List[Any]:
    """Replace missing values (None, '', and NaN) with a given filling value.

    Args:
        values: Input list potentially containing missing values.
        fill_value: Value used to replace missing entries. Defaults to 0.

    Returns:
        List of values with missing entries replaced by fill_value.
    """
    return [fill_value if _is_missing(value) else value for value in values]


def remove_duplicates(values: List[Any]) -> List[Any]:
    """Remove duplicate values from a list while preserving order.

    Args:
        values: Input list of values.

    Returns:
        List of unique values in their original order of first appearance.
    """
    unique_values: List[Any] = []
    seen_hashable: set[Any] = set()
    for item in values:
        try:
            if item not in seen_hashable:
                seen_hashable.add(item)
                unique_values.append(item)
        except TypeError:
            if item not in unique_values:
                unique_values.append(item)
    return unique_values


def min_max_normalize(
    values: List[float],
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> List[float]:
    """Normalize numerical values to a target range using the min-max method.

    Args:
        values: List of numerical values.
        new_min: Lower bound of the normalized range. Defaults to 0.0.
        new_max: Upper bound of the normalized range. Defaults to 1.0.

    Returns:
        List of normalized numerical values.

    Raises:
        ValueError: If new_min is greater than new_max.
    """
    if new_min > new_max:
        raise ValueError("new_min must be less than or equal to new_max.")
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [float(new_min) for _ in values]

    value_range = max_val - min_val
    target_range = new_max - new_min
    return [new_min + ((v - min_val) / value_range) * target_range for v in values]


def z_score_standardize(values: List[float]) -> List[float]:
    """Standardize numerical values using the z-score method.

    Args:
        values: List of numerical values.

    Returns:
        List of standardized values with zero mean and unit variance.
    """
    if not values:
        return []

    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_dev = math.sqrt(variance)

    if std_dev == 0.0:
        return [0.0 for _ in values]

    return [(v - mean_val) / std_dev for v in values]


def clip_values(
    values: List[float],
    min_val: float,
    max_val: float,
) -> List[float]:
    """Clip numerical values to the inclusive range [min_val, max_val].

    Args:
        values: List of numerical values.
        min_val: Minimum value to clip to.
        max_val: Maximum value to clip to.

    Returns:
        List of clipped numerical values.

    Raises:
        ValueError: If min_val is greater than max_val.
    """
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val.")
    return [max(min_val, min(max_val, v)) for v in values]


def convert_to_integers(values: List[str]) -> List[int]:
    """Convert numerical strings in a list to integers, excluding non-numerical ones.

    Args:
        values: List of strings containing numerical and/or non-numerical values.

    Returns:
        List of integers converted from valid numerical strings.
    """
    result: List[int] = []
    for item in values:
        try:
            result.append(int(item))
        except (ValueError, TypeError):
            try:
                float_val = float(item)
                if math.isfinite(float_val):
                    result.append(int(float_val))
            except (ValueError, TypeError):
                continue
    return result


def log_transform(values: List[float]) -> List[float]:
    """Apply natural logarithmic scale transformation to positive numbers.

    Non-positive numbers, NaN, infinite, and non-numeric values are excluded.

    Args:
        values: List of numerical values.

    Returns:
        List of natural logarithm values for the strictly positive entries.
    """
    return [
        math.log(v)
        for v in values
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v > 0
    ]


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words.

    Args:
        text: Input string to process.

    Returns:
        List of lowercased alphanumeric word tokens.
    """
    if not text:
        return []
    return re.findall(r"[^\W_]+", text.lower())


def select_alphanumeric_and_spaces(text: str) -> str:
    """Keep only alphanumeric characters and spaces from text.

    Args:
        text: Input string to process.

    Returns:
        Processed string containing only alphanumeric characters and spaces.
    """
    return "".join(char for char in text if char.isalnum() or char == " ")


# Alias to match the assignment wording
remove_punctuation = select_alphanumeric_and_spaces


def remove_stop_words(text: str, stop_words: Iterable[str]) -> str:
    """Remove stop-words from text after lowercasing it.

    Args:
        text: Input text to process.
        stop_words: Collection of stop-words to remove.

    Returns:
        Lowercased text with stop-words removed.
    """
    if not text:
        return ""
    stop_set = {word.lower() for word in stop_words}
    words = text.lower().split()
    filtered = [word for word in words if word not in stop_set]
    return " ".join(filtered)


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Flatten a list of lists into a single list.

    Args:
        nested_list: List containing nested lists (or elements).

    Returns:
        A single flattened list.
    """
    flattened: List[Any] = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def shuffle_list(values: List[Any], seed: Optional[int] = None) -> List[Any]:
    """Return a randomly shuffled copy of a list using an optional seed.

    Args:
        values: Input list of values.
        seed: Optional seed to ensure reproducibility.

    Returns:
        A new list containing the shuffled values.
    """
    shuffled = list(values)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled
