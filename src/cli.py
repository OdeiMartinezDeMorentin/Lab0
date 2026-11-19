"""Command Line Interface (CLI) implementation for the MLOps project."""

import json
from typing import Any, List, Optional

import click

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
    shuffle_list,
    tokenize_text,
    z_score_standardize,
)


def _parse_value(raw: str) -> Any:
    """Parse a raw CLI token into None, float('nan'), empty string, or original value."""
    if raw == "None":
        return None
    if raw.lower() == "nan":
        return float("nan")
    return raw


@click.group()
@click.version_option(version="0.1.0", prog_name="mlops-cli")
def cli() -> None:
    """MLOps command line interface for data preprocessing."""


# ---------------------------------------------------------------------------
# Group 1: Clean
# ---------------------------------------------------------------------------
@cli.group()
def clean() -> None:
    """Data cleaning commands (handling missing values)."""


@clean.command("remove-missing")
@click.argument("values", nargs=-1)
def remove_missing_cmd(values: tuple[str, ...]) -> None:
    """Remove missing values (None, '', nan) from VALUES.

    \b
    Example:
      python -m src.cli clean remove-missing 1 None "" nan 2
    """
    parsed = [_parse_value(v) for v in values]
    click.echo(remove_missing_values(parsed))


@clean.command("fill-missing")
@click.argument("values", nargs=-1)
@click.option(
    "--fill-value",
    default="0",
    show_default=True,
    help="Value used to fill missing entries.",
)
def fill_missing_cmd(values: tuple[str, ...], fill_value: str) -> None:
    """Replace missing values in VALUES with FILL_VALUE.

    \b
    Example:
      python -m src.cli clean fill-missing 1 None "" nan 2 --fill-value 0
    """
    parsed = [_parse_value(v) for v in values]
    click.echo(fill_missing_values(parsed, fill_value=fill_value))


# ---------------------------------------------------------------------------
# Group 2: Numeric
# ---------------------------------------------------------------------------
@cli.group()
def numeric() -> None:
    """Numerical preprocessing commands."""


@numeric.command("normalize")
@click.argument("values", nargs=-1, type=float)
@click.option(
    "--new-min",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum value of the target range.",
)
@click.option(
    "--new-max",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum value of the target range.",
)
def normalize_cmd(values: tuple[float, ...], new_min: float, new_max: float) -> None:
    """Normalize numerical VALUES using the min-max method.

    \b
    Example:
      python -m src.cli numeric normalize 0 5 10 --new-min 0.0 --new-max 1.0
    """
    click.echo(min_max_normalize(list(values), new_min=new_min, new_max=new_max))


@numeric.command("standardize")
@click.argument("values", nargs=-1, type=float)
def standardize_cmd(values: tuple[float, ...]) -> None:
    """Standardize numerical VALUES using the z-score method.

    \b
    Example:
      python -m src.cli numeric standardize 10 20 30
    """
    click.echo(z_score_standardize(list(values)))


@numeric.command("clip")
@click.argument("values", nargs=-1, type=float)
@click.option(
    "--min-val",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum value to clip to.",
)
@click.option(
    "--max-val",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum value to clip to.",
)
def clip_cmd(values: tuple[float, ...], min_val: float, max_val: float) -> None:
    """Clip numerical VALUES to [min_val, max_val].

    \b
    Example:
      python -m src.cli numeric clip --min-val 0 --max-val 10 -- -5 3 15
    """
    click.echo(clip_values(list(values), min_val=min_val, max_val=max_val))


@numeric.command("to-int")
@click.argument("values", nargs=-1)
def to_int_cmd(values: tuple[str, ...]) -> None:
    """Convert numerical string VALUES to integers, excluding non-numerical ones.

    \b
    Example:
      python -m src.cli numeric to-int 10 abc 20 -5 3.14
    """
    click.echo(convert_to_integers(list(values)))


@numeric.command("log")
@click.argument("values", nargs=-1, type=float)
def log_cmd(values: tuple[float, ...]) -> None:
    """Apply natural logarithmic transformation to positive numerical VALUES.

    \b
    Example:
      python -m src.cli numeric log 1 2.71828 7.389
    """
    click.echo(log_transform(list(values)))


# ---------------------------------------------------------------------------
# Group 3: Text
# ---------------------------------------------------------------------------
@cli.group()
def text() -> None:
    """Text preprocessing commands."""


@text.command("tokenize")
@click.argument("input_text")
def tokenize_cmd(input_text: str) -> None:
    """Tokenize INPUT_TEXT into lowercased alphanumeric words.

    \b
    Example:
      python -m src.cli text tokenize "Hello, World! 2026"
    """
    click.echo(tokenize_text(input_text))


@text.command("remove-punctuation")
@click.argument("input_text")
def remove_punctuation_cmd(input_text: str) -> None:
    """Select alphanumeric characters and spaces from INPUT_TEXT, removing punctuation.

    \b
    Example:
      python -m src.cli text remove-punctuation "Hello, #MLOps_2026!"
    """
    click.echo(remove_punctuation(input_text))


@text.command("remove-stopwords")
@click.argument("input_text")
@click.option(
    "--stop-word",
    "stop_words",
    multiple=True,
    help="Stop-word to remove (can be specified multiple times).",
)
def remove_stopwords_cmd(input_text: str, stop_words: tuple[str, ...]) -> None:
    """Remove STOP_WORDS from lowercased INPUT_TEXT.

    \b
    Example:
      python -m src.cli text remove-stopwords "The quick Fox" --stop-word the --stop-word is
    """
    click.echo(remove_stop_words(input_text, stop_words))


# ---------------------------------------------------------------------------
# Group 4: Struct
# ---------------------------------------------------------------------------
@cli.group()
def struct() -> None:
    """List structure preprocessing commands."""


@struct.command("shuffle")
@click.argument("values", nargs=-1)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Seed to ensure reproducibility.",
)
def shuffle_cmd(values: tuple[str, ...], seed: Optional[int]) -> None:
    """Randomly shuffle VALUES using an optional SEED.

    \b
    Example:
      python -m src.cli struct shuffle a b c d --seed 42
    """
    click.echo(shuffle_list(list(values), seed=seed))


@struct.command("flatten")
@click.argument("nested_json")
def flatten_cmd(nested_json: str) -> None:
    """Flatten a JSON-encoded list of lists (NESTED_JSON).

    \b
    Example:
      python -m src.cli struct flatten "[[1, 2], [3, 4]]"
    """
    parsed: List[Any] = json.loads(nested_json)
    click.echo(flatten_list(parsed))


@struct.command("unique-values")
@click.argument("values", nargs=-1)
def unique_values_cmd(values: tuple[str, ...]) -> None:
    """Remove duplicate entries from VALUES, returning unique values.

    \b
    Example:
      python -m src.cli struct unique-values a b a c 1 2 1
    """
    click.echo(remove_duplicates(list(values)))


if __name__ == "__main__":  # pragma: no cover
    cli()
