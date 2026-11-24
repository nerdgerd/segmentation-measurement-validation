"""Keep default test scratch files inside the commissioned study workspace."""

import os
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        workspace = Path(__file__).resolve().parents[1]
        parent = (workspace / "cache/test-scratch").resolve()
        target = (parent / f"pytest-{os.getpid()}").resolve()
        if not parent.is_relative_to(workspace) or not target.is_relative_to(parent):
            raise ValueError("Default pytest scratch path escaped the study workspace")
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(target)
