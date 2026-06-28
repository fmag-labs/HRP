"""Test isolation for ops threshold tests.

`load_thresholds()` applies `HRP_THRESHOLD_*` environment overrides on top of any
YAML/defaults. Those vars leak into ``os.environ`` whenever a module that calls
``load_dotenv()`` (e.g. ``hrp.utils.config``) is imported by an earlier test, so
ambient values from the developer's ``.env`` can bleed in and make the threshold
tests order-dependent. Clear them before each test so these tests are hermetic;
tests that set their own override via ``monkeypatch.setenv`` still work.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_threshold_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("HRP_THRESHOLD_"):
            monkeypatch.delenv(key, raising=False)
