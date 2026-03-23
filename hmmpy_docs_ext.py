from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def setup(app: Any) -> dict[str, bool]:
    root = Path(__file__).resolve().parent
    src = root / "src"
    jupyter_data_dir = root / "_build" / ".jupyter"
    kernel_dir = jupyter_data_dir / "kernels" / "hmmpy-docs"

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    python_path = os.environ.get("PYTHONPATH")
    python_path = str(src) if not python_path else f"{src}{os.pathsep}{python_path}"

    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-Xfrozen_modules=off",
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "HMMPY Docs",
                "language": "python",
                "metadata": {"debugger": True},
                "env": {"PYTHONPATH": python_path},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.environ["PYTHONPATH"] = python_path
    existing_jupyter_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = (
        str(jupyter_data_dir)
        if not existing_jupyter_path
        else f"{jupyter_data_dir}{os.pathsep}{existing_jupyter_path}"
    )

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
