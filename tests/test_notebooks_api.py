"""Static checks for notebook example API usage."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def test_notebooks_do_not_use_removed_scaling_flag() -> None:
    """Notebook examples should use ComputeMode instead of the removed scaling flag."""
    stale_references: list[str] = []

    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell.get("source", []))
            if "scaling=" in source:
                stale_references.append(f"{notebook_path.name}:cell{index}")

    assert stale_references == []
