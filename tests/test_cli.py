"""Integration and CLI tests for the Click interface."""

from click.testing import CliRunner

from src.cli import cli


def test_cli_help_and_version() -> None:
    """Test main CLI help and version options."""
    runner = CliRunner()
    help_res = runner.invoke(cli, ["--help"])
    assert help_res.exit_code == 0
    assert "clean" in help_res.output
    assert "numeric" in help_res.output
    assert "text" in help_res.output
    assert "struct" in help_res.output

    ver_res = runner.invoke(cli, ["--version"])
    assert ver_res.exit_code == 0
    assert "0.1.0" in ver_res.output


def test_cli_clean_commands() -> None:
    """Test clean command group: remove-missing, fill-missing, remove-duplicates."""
    runner = CliRunner()

    res_remove = runner.invoke(
        cli, ["clean", "remove-missing", "1", "None", "", "nan", "2"]
    )
    assert res_remove.exit_code == 0
    assert res_remove.output.strip() == "['1', '2']"

    res_fill = runner.invoke(
        cli, ["clean", "fill-missing", "1", "None", "", "nan", "--fill-value", "0"]
    )
    assert res_fill.exit_code == 0
    assert res_fill.output.strip() == "['1', '0', '0', '0']"

    res_dedup = runner.invoke(cli, ["clean", "remove-duplicates", "a", "b", "a", "c"])
    assert res_dedup.exit_code == 0
    assert res_dedup.output.strip() == "['a', 'b', 'c']"


def test_cli_numeric_commands() -> None:
    """Test numeric command group: normalize, standardize, clip, to-int, log."""
    runner = CliRunner()

    res_norm = runner.invoke(cli, ["numeric", "normalize", "0", "5", "10"])
    assert res_norm.exit_code == 0
    assert res_norm.output.strip() == "[0.0, 0.5, 1.0]"

    res_std = runner.invoke(cli, ["numeric", "standardize", "5", "5"])
    assert res_std.exit_code == 0
    assert res_std.output.strip() == "[0.0, 0.0]"

    res_clip = runner.invoke(
        cli,
        ["numeric", "clip", "--min-val", "0", "--max-val", "5", "--", "-2", "3", "8"],
    )
    assert res_clip.exit_code == 0
    assert res_clip.output.strip() == "[0.0, 3.0, 5.0]"

    res_int = runner.invoke(cli, ["numeric", "to-int", "10", "abc", "20"])
    assert res_int.exit_code == 0
    assert res_int.output.strip() == "[10, 20]"

    res_log = runner.invoke(cli, ["numeric", "log", "--", "-1", "0", "1"])
    assert res_log.exit_code == 0
    assert res_log.output.strip() == "[0.0]"


def test_cli_text_commands() -> None:
    """Test text command group: tokenize, alphanumeric, remove-stopwords."""
    runner = CliRunner()

    res_tok = runner.invoke(cli, ["text", "tokenize", "Hello, World! 123"])
    assert res_tok.exit_code == 0
    assert res_tok.output.strip() == "['hello', 'world', '123']"

    res_alnum = runner.invoke(cli, ["text", "alphanumeric", "Hello, #MLOps!"])
    assert res_alnum.exit_code == 0
    assert res_alnum.output.strip() == "Hello MLOps"

    res_stop = runner.invoke(
        cli,
        ["text", "remove-stopwords", "The quick Fox", "--stop-word", "the"],
    )
    assert res_stop.exit_code == 0
    assert res_stop.output.strip() == "quick fox"


def test_cli_struct_commands() -> None:
    """Test struct command group: flatten and shuffle."""
    runner = CliRunner()

    res_flat = runner.invoke(cli, ["struct", "flatten", "[[1, 2], [3, 4]]"])
    assert res_flat.exit_code == 0
    assert res_flat.output.strip() == "[1, 2, 3, 4]"

    res_shuf1 = runner.invoke(
        cli, ["struct", "shuffle", "a", "b", "c", "d", "--seed", "42"]
    )
    res_shuf2 = runner.invoke(
        cli, ["struct", "shuffle", "a", "b", "c", "d", "--seed", "42"]
    )
    assert res_shuf1.exit_code == 0
    assert res_shuf1.output == res_shuf2.output
