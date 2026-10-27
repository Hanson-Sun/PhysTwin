"""Run directories and manifests for grouped-FIM experiments.

Every run lives under ``<root>/<case>/<run_id>/`` and records its config, the
arrays it wrote, and a manifest that can be reloaded without guessing shapes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

MANIFEST_VERSION = 1


def progress_printer(started: float) -> Callable[[int, int], None]:
    """Print progress and an ETA for a loop of rollouts."""

    def report(done: int, total: int) -> None:
        elapsed = time.time() - started
        print(
            f"[{done:3d}/{total:3d}] {elapsed / 60:6.1f} min, "
            f"eta {(total - done) * elapsed / done / 60:6.1f} min",
            flush=True,
        )

    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass
class Run:
    root: Path
    case: str
    run_id: str
    config: dict[str, Any]
    arrays: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        root: str | Path,
        case: str,
        run_id: str,
        config: dict[str, Any],
    ) -> "Run":
        for value in (case, run_id):
            if not value or Path(value).name != value:
                raise ValueError(f"case and run_id must be single path components: {value!r}")
        run = cls(Path(root).resolve(), case, run_id, config)
        run.path.mkdir(parents=True, exist_ok=True)
        run._write_json("config.json", config)
        return run

    @property
    def path(self) -> Path:
        return self.root / self.case / self.run_id

    def save_array(self, name: str, values: Any, role: str) -> Path:
        """Write an array and record its role, shape and dtype."""
        array = np.asarray(values)
        path = self._resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        self.arrays[path.relative_to(self.path).as_posix()] = {
            "role": role,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        return path

    def save_json(self, name: str, payload: Any) -> Path:
        return self._write_json(name, payload)

    def close(self) -> Path:
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "case": self.case,
            "run_id": self.run_id,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": "config.json",
            "arrays": self.arrays,
        }
        self._write_json("manifest.json", manifest)
        return self.path / "manifest.json"

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _resolve(self, name: str) -> Path:
        path = (self.path / name).resolve()
        if path != self.path.resolve() and self.path.resolve() not in path.parents:
            raise ValueError(f"path escapes run directory: {name!r}")
        return path

    def _write_json(self, name: str, payload: Any) -> Path:
        path = self._resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=_json_default)
        return path
