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
    remove_stop_words,
    select_alphanumeric_and_spaces,
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


@cli.group()
def clean() -> None:
    """Data cleaning commands (missing values and duplicates)."""


@clean.command("remove-missing")
@click.argument("values", nargs=-1)
def remove_missing_cmd(values: tuple[str, ...]) -> None:
    """Remove missing values (None, '', nan) from VALUES."""
    parsed = [_parse_value(v) for v in values]
    click.echo(remove_missing_values(parsed))


@clean.command("fill-missing")
@click.argument("values", nargs=-1)
@click.option("--fill-value", default="0", help="Value used to fill missing entries.")
def fill_missing_cmd(values: tuple[str, ...], fill_value: str) -> None:
    """Replace missing values in VALUES with FILL_VALUE."""
    parsed = [_parse_value(v) for v in values]
    click.echo(fill_missing_values(parsed, fill_value=fill_value))


@clean.command("remove-duplicates")
@click.argument("values", nargs=-1)
def remove_duplicates_cmd(values: tuple[str, ...]) -> None:
    """Remove duplicate entries from VALUES."""
    click.echo(remove_duplicates(list(values)))


@cli.group()
def numeric() -> None:
    """Numerical preprocessing commands."""


@numeric.command("normalize")
@click.argument("values", nargs=-1, type=float)
@click.option(
    "--new-min", default=0.0, type=float, help="Minimum value of target range."
)
@click.option(
    "--new-max", default=1.0, type=float, help="Maximum value of target range."
)
def normalize_cmd(values: tuple[float, ...], new_min: float, new_max: float) -> None:
    """Normalize numerical VALUES using the min-max method."""
    click.echo(min_max_normalize(list(values), new_min=new_min, new_max=new_max))


@numeric.command("standardize")
@click.argument("values", nargs=-1, type=float)
def standardize_cmd(values: tuple[float, ...]) -> None:
    """Standardize numerical VALUES using the z-score method."""
    click.echo(z_score_standardize(list(values)))


@numeric.command("clip")
@click.argument("values", nargs=-1, type=float)
@click.option("--min-val", required=True, type=float, help="Minimum value to clip to.")
@click.option("--max-val", required=True, type=float, help="Maximum value to clip to.")
def clip_cmd(values: tuple[float, ...], min_val: float, max_val: float) -> None:
    """Clip numerical VALUES to [min_val, max_val]."""
    click.echo(clip_values(list(values), min_val=min_val, max_val=max_val))


@numeric.command("to-int")
@click.argument("values", nargs=-1)
def to_int_cmd(values: tuple[str, ...]) -> None:
    """Convert numerical string VALUES to integers, excluding non-numerical ones."""
    click.echo(convert_to_integers(list(values)))


@numeric.command("log")
@click.argument("values", nargs=-1, type=float)
def log_cmd(values: tuple[float, ...]) -> None:
    """Apply natural logarithmic transformation to positive numerical VALUES."""
    click.echo(log_transform(list(values)))


@cli.group()
def text() -> None:
    """Text preprocessing commands."""


@text.command("tokenize")
@click.argument("input_text")
def tokenize_cmd(input_text: str) -> None:
    """Tokenize INPUT_TEXT into lowercased alphanumeric words."""
    click.echo(tokenize_text(input_text))


@text.command("alphanumeric")
@click.argument("input_text")
def alphanumeric_cmd(input_text: str) -> None:
    """Keep only alphanumeric characters and spaces from INPUT_TEXT."""
    click.echo(select_alphanumeric_and_spaces(input_text))


@text.command("remove-stopwords")
@click.argument("input_text")
@click.option(
    "--stop-word",
    "stop_words",
    multiple=True,
    help="Stop-word to remove (can be specified multiple times).",
)
def remove_stopwords_cmd(input_text: str, stop_words: tuple[str, ...]) -> None:
    """Remove STOP_WORDS from lowercased INPUT_TEXT."""
    click.echo(remove_stop_words(input_text, stop_words))


@cli.group()
def struct() -> None:
    """List structure preprocessing commands."""


@struct.command("flatten")
@click.argument("nested_json")
def flatten_cmd(nested_json: str) -> None:
    """Flatten a JSON-encoded list of lists (NESTED_JSON)."""
    parsed: List[Any] = json.loads(nested_json)
    click.echo(flatten_list(parsed))


@struct.command("shuffle")
@click.argument("values", nargs=-1)
@click.option("--seed", default=None, type=int, help="Seed for reproducibility.")
def shuffle_cmd(values: tuple[str, ...], seed: Optional[int]) -> None:
    """Randomly shuffle VALUES using an optional SEED."""
    click.echo(shuffle_list(list(values), seed=seed))


@clean.command("remove-duplicates")
@click.argument("values", nargs=-1)
def remove_duplicates_cmd(values: tuple[str, ...]) -> None:
    """Remove duplicate entries from VALUES."""
    click.echo(remove_duplicates(list(values)))


@cli.group()
def numeric() -> None:
    """Numerical preprocessing commands."""


@numeric.command("normalize")
@click.argument("values", nargs=-1, type=float)
@click.option("--new-min", default=0.0, type=float, help="Minimum value of target range.")
@click.option("--new-max", default=1.0, type=float, help="Maximum value of target range.")
def normalize_cmd(values: tuple[float, ...], new_min: float, new_max: float) -> None:
    """Normalize numerical VALUES using the min-max method."""
    click.echo(min_max_normalize(list(values), new_min=new_min, new_max=new_max))


@numeric.command("standardize")
@click.argument("values", nargs=-1, type=float)
def standardize_cmd(values: tuple[float, ...]) -> None:
    """Standardize numerical VALUES using the z-score method."""
    click.echo(z_score_standardize(list(values)))


@numeric.command("clip")
@click.argument("values", nargs=-1, type=float)
@click.option("--min-val", required=True, type=float, help="Minimum value to clip to.")
@click.option("--max-val", required=True, type=float, help="Maximum value to clip to.")
def clip_cmd(values: tuple[float, ...], min_val: float, max_val: float) -> None:
    """Clip numerical VALUES to [min_val, max_val]."""
    click.echo(clip_values(list(values), min_val=min_val, max_val=max_val))


@numeric.command("to-int")
@click.argument("values", nargs=-1)
def to_int_cmd(values: tuple[str, ...]) -> None:
    """Convert numerical string VALUES to integers, excluding non-numerical ones."""
    click.echo(convert_to_integers(list(values)))


@numeric.command("log")
@click.argument("values", nargs=-1, type=float)
def log_cmd(values: tuple[float, ...]) -> None:
    """Apply natural logarithmic transformation to positive numerical VALUES."""
    click.echo(log_transform(list(values)))


@cli.group()
def text() -> None:
    """Text preprocessing commands."""


@text.command("tokenize")
@click.argument("input_text")
def tokenize_cmd(input_text: str) -> None:
    """Tokenize INPUT_TEXT into lowercased alphanumeric words."""
    click.echo(tokenize_text(input_text))


@text.command("alphanumeric")
@click.argument("input_text")
def alphanumeric_cmd(input_text: str) -> None:
    """Keep only alphanumeric characters and spaces from INPUT_TEXT."""
    click.echo(select_alphanumeric_and_spaces(input_text))


@text.command("remove-stopwords")
@click.argument("input_text")
@click.option(
    "--stop-word",
    "stop_words",
    multiple=True,
    help="Stop-word to remove (can be specified multiple times).",
)
def remove_stopwords_cmd(input_text: str, stop_words: tuple[str, ...]) -> None:
    """Remove STOP_WORDS from lowercased INPUT_TEXT."""
    click.echo(remove_stop_words(input_text, stop_words))


@cli.group()
def struct() -> None:
    """List structure preprocessing commands."""


@struct.command("flatten")
@click.argument("nested_json")
def flatten_cmd(nested_json: str) -> None:
    """Flatten a JSON-encoded list of lists (NESTED_JSON)."""
    parsed: List[Any] = json.loads(nested_json)
    click.echo(flatten_list(parsed))


@struct.command("shuffle")
@click.argument("values", nargs=-1)
@click.option("--seed", default=None, type=int, help="Seed for reproducibility.")
def shuffle_cmd(values: tuple[str, ...], seed: Optional[int]) -> None:
    """Randomly shuffle VALUES using an optional SEED."""
    click.echo(shuffle_list(list(values), seed=seed))


if __name__ == "__main__":  # pragma: no cover
    cli()
