"""Output helpers shared by clean runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def scalarize_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Return JSON/CSV-friendly scalar fields from an estimator or result dict."""

    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (pd.Series, pd.DataFrame, list, tuple, dict)):
            continue
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def write_csv_json(
    frame: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a DataFrame and compact JSON summary to ``output_dir``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}.csv"
    json_path = out / f"{stem}_summary.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        **(summary or {}),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return csv_path, json_path


__all__ = ["scalarize_mapping", "write_csv_json"]
