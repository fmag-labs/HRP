#!/usr/bin/env python3
"""Auto-bump the project version using CalVer ``YYYY.MMDD.MICRO``.

You never specify the number: the date comes from the clock and ``MICRO``
auto-increments when there is already a release for today. Run this as the first
step of a release (the assistant does this automatically when opening a PR).

    python scripts/bump_version.py            # bump pyproject + stamp CHANGELOG
    python scripts/bump_version.py --print     # print the next version, no writes
    python scripts/bump_version.py --current   # print the current version

``MMDD`` drops the leading zero (PEP 440 form): June 28 -> ``628`` (so today is
``2026.628.0``), Jan 5 -> ``105``. The single source of truth is
``pyproject.toml``; ``hrp.__version__`` reads that literal.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
WEB_PACKAGE = ROOT / "web" / "package.json"

# Match the version on the first `version = "..."` line of [project].
_VERSION_RE = re.compile(r'^version = "([^"]+)"', re.MULTILINE)
# Match the "version": "..." field in package.json.
_PKG_VERSION_RE = re.compile(r'("version":\s*")[^"]+(")')


def current_version() -> str:
    match = _VERSION_RE.search(PYPROJECT.read_text())
    if not match:
        raise SystemExit("error: could not find `version` in pyproject.toml")
    return match.group(1)


def next_version(today: date | None = None) -> str:
    """Today's CalVer, incrementing MICRO past any existing same-day release."""
    today = today or date.today()
    # MMDD with no leading zero (PEP 440 form): June 28 -> 628, Jan 5 -> 105.
    prefix = f"{today.year}.{today.month * 100 + today.day}"
    current = current_version()
    micro = 0
    if current.startswith(prefix + "."):
        try:
            micro = int(current.rsplit(".", 1)[1]) + 1
        except ValueError:
            micro = 0
    return f"{prefix}.{micro}"


def _stamp_changelog(version: str) -> None:
    if not CHANGELOG.exists():
        return
    text = CHANGELOG.read_text()
    header = f"## [{version}] - {date.today().isoformat()}"
    if header in text:
        return
    marker = "## [Unreleased]"
    if marker in text:
        # Open a release section right under [Unreleased]; it inherits the
        # accumulated unreleased notes, leaving a fresh empty [Unreleased] on top.
        text = text.replace(marker, f"{marker}\n\n{header}", 1)
        CHANGELOG.write_text(text)


def _bump_web(version: str) -> None:
    """Keep the Next.js app (web/package.json) version in sync."""
    if not WEB_PACKAGE.exists():
        return
    text = WEB_PACKAGE.read_text()
    new_text, count = _PKG_VERSION_RE.subn(rf"\g<1>{version}\g<2>", text, count=1)
    if count:
        WEB_PACKAGE.write_text(new_text)


def bump() -> str:
    new = next_version()
    PYPROJECT.write_text(_VERSION_RE.sub(f'version = "{new}"', PYPROJECT.read_text(), count=1))
    _bump_web(new)
    _stamp_changelog(new)
    return new


if __name__ == "__main__":
    if "--current" in sys.argv:
        print(current_version())
    elif "--print" in sys.argv:
        print(next_version())
    else:
        print(bump())
