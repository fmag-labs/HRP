"""Test CLI entrypoint."""


def test_cli_module_imports():
    """CLI module should import without error."""
    from hrp import cli
    assert hasattr(cli, "main")


def test_cli_main_is_callable():
    """CLI main should be callable."""
    from hrp.cli import main
    assert callable(main)
