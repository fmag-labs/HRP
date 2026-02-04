"""Test version alignment."""


def test_version_is_string():
    """Version should be a string."""
    import hrp
    assert isinstance(hrp.__version__, str)


def test_version_matches_pyproject():
    """Version should match pyproject.toml."""
    import hrp
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    expected = data["project"]["version"]
    assert hrp.__version__ == expected
