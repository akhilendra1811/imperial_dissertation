#!/usr/bin/env python3
"""
Stream annual LOBSTER .7z archives through the project LOBSTER preprocessor.
The compressed data files are saved on my one-drive and this will allow me to store
alot of the files without take much space.

This runner is intended for large annual LOBSTER downloads. It avoids extracting
a full archive to disk before processing. Instead, archive members are streamed,
daily message/orderbook pairs are temporarily materialised, and the numerical
aggregation is done by preprocessing/preprocess_lobster_intraday.py.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gzip
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pandas as pd


# libarchive status codes.
ARCHIVE_OK = 0
ARCHIVE_EOF = 1

# Read archive data in 1 MiB chunks. 
BUFFER_SIZE = 1024 * 1024


def find_project_root() -> Path:
    """Infer the project root from the location of this runner.
    """
    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        if parent.name in {"runners", "preprocessing"}:
            return parent.parent

    return current_file.parent


PROJECT_ROOT = find_project_root()
DEFAULT_PREPROCESSOR = PROJECT_ROOT / "preprocessing" / "preprocess_lobster_intraday.py"


def load_project_preprocessor(path: Path) -> ModuleType:
    """Load the existing LOBSTER preprocessor from a file path.

    The streaming runner does not duplicate the numerical
    aggregation logic. It imports the project preprocessor and calls:

        parse_lobster_filename
        LobsterDay
        aggregate_lobster_day
        append_csv_gz
    """
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file does not exist: {path}")

    spec = importlib.util.spec_from_file_location("project_lobster_preprocessor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import preprocessor: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LibArchiveReader:
    """Small context-manager wrapper around system libarchive.

    Python's standard library cannot read .7z archives. This wrapper uses the
    system libarchive library through ctypes so that archive entries can be
    streamed without extracting the full archive first.
    """

    def __init__(self, archive_path: Path) -> None:
        library_path = ctypes.util.find_library("archive")
        if not library_path:
            raise RuntimeError(
                "libarchive was not found. Install libarchive or run this on a system "
                "where the archive library is available."
            )

        self.lib = ctypes.CDLL(library_path)
        self._configure_ctypes_signatures()

        self.handle = self.lib.archive_read_new()
        if not self.handle:
            raise RuntimeError("archive_read_new failed")

        self.lib.archive_read_support_filter_all(self.handle)
        self.lib.archive_read_support_format_all(self.handle)

        result = self.lib.archive_read_open_filename(
            self.handle,
            str(archive_path).encode("utf-8"),
            BUFFER_SIZE,
        )
        if result != ARCHIVE_OK:
            error = self.lib.archive_error_string(self.handle)
            self.close()
            raise RuntimeError(
                f"Could not open {archive_path}: "
                f"{error.decode('utf-8', errors='replace') if error else result}"
            )

    def _configure_ctypes_signatures(self) -> None:
        """Declare libarchive function signatures used by ctypes.

        Without these signatures, ctypes may make unsafe assumptions about return
        types and argument types.
        """
        lib = self.lib

        lib.archive_read_new.restype = ctypes.c_void_p

        lib.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
        lib.archive_read_support_format_all.argtypes = [ctypes.c_void_p]

        lib.archive_read_open_filename.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        lib.archive_read_next_header.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
        lib.archive_entry_pathname.restype = ctypes.c_char_p

        lib.archive_read_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.archive_read_data.restype = ctypes.c_ssize_t

        lib.archive_error_string.argtypes = [ctypes.c_void_p]
        lib.archive_error_string.restype = ctypes.c_char_p

        lib.archive_read_free.argtypes = [ctypes.c_void_p]

    def entries(self):
        """Yield archive entries as ``(member_name, chunks)``.

        The ``chunks`` object is a generator and must be consumed before moving
        to the next entry. This keeps archive reading sequential and memory-light.
        """
        entry = ctypes.c_void_p()
        buffer = ctypes.create_string_buffer(BUFFER_SIZE)

        while True:
            result = self.lib.archive_read_next_header(self.handle, ctypes.byref(entry))

            if result == ARCHIVE_EOF:
                break

            if result != ARCHIVE_OK:
                error = self.lib.archive_error_string(self.handle)
                raise RuntimeError(
                    error.decode("utf-8", errors="replace") if error else result
                )

            raw_name = self.lib.archive_entry_pathname(entry)
            name = raw_name.decode("utf-8") if raw_name else ""

            def chunks():
                while True:
                    size = self.lib.archive_read_data(self.handle, buffer, BUFFER_SIZE)

                    if size == 0:
                        break

                    if size < 0:
                        error = self.lib.archive_error_string(self.handle)
                        raise RuntimeError(
                            error.decode("utf-8", errors="replace") if error else size
                        )

                    yield buffer.raw[:size]

            yield name, chunks()

    def close(self) -> None:
        """Release the libarchive handle."""
        if getattr(self, "handle", None):
            self.lib.archive_read_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def extract_entry(destination: Path, chunks) -> None:
    """Write one streamed archive member to disk.

    The project preprocessor expects file paths, not in-memory file objects, so
    each matched message/orderbook file is temporarily materialised.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as output:
        for chunk in chunks:
            output.write(chunk)


def estimate_existing_rows(ticker_daily: Path) -> tuple[int, int]:
    """Return ``(days, rows)`` for a previously processed ticker.

    Older quality files may not contain ``minute_rows``. In that case, fall back
    to 390 rows per day, which is the regular US trading-day minute count.
    """
    existing = pd.read_csv(ticker_daily)
    days = int(len(existing))

    if "minute_rows" in existing.columns:
        rows = int(existing["minute_rows"].sum())
    else:
        rows = int(days * 390)

    return days, rows


def process_archive(
    archive: Path,
    output_dir: Path,
    preprocessor: ModuleType,
    force: bool = False,
) -> dict:
    """Process one annual ticker archive.

    Each completed ticker gets two per-ticker files:
        - compressed minute panel
        - daily quality summary

    Writing first goes to ``.part`` files. These are atomically renamed only
    after successful completion, which avoids leaving corrupted final outputs if
    the run is interrupted.
    """
    ticker = archive.name.split("_", 1)[0]

    ticker_panel = output_dir / "per_ticker" / f"{ticker}_minute_prices.csv.gz"
    ticker_daily = output_dir / "per_ticker" / f"{ticker}_daily_quality.csv"

    if not force and ticker_panel.exists() and ticker_daily.exists():
        days, rows = estimate_existing_rows(ticker_daily)
        print(f"{ticker}: already complete ({days} days)", flush=True)

        return {
            "ticker": ticker,
            "days": days,
            "rows": rows,
            "panel": str(ticker_panel),
            "daily_quality": str(ticker_daily),
            "skipped_existing": True,
        }

    ticker_panel.parent.mkdir(parents=True, exist_ok=True)

    panel_part = ticker_panel.with_suffix(ticker_panel.suffix + ".part")
    daily_part = ticker_daily.with_suffix(ticker_daily.suffix + ".part")

    panel_part.unlink(missing_ok=True)
    daily_part.unlink(missing_ok=True)

    summaries: list[dict] = []
    rows_written = 0

    # Holds unmatched message/orderbook files until their counterpart appears.
    # In usual LOBSTER archive order this should stay small. If an archive stores
    # all message files before all orderbook files, more temporary files may be
    # held at once, but the full archive is still not extracted.
    pending: dict[tuple[str, str, str, str, str], dict[str, Path]] = {}

    temp_root = Path(tempfile.mkdtemp(prefix=f"lobster_{ticker}_"))

    try:
        with LibArchiveReader(archive) as reader:
            for member_name, chunks in reader.entries():
                # Use only the basename to avoid writing unsafe archive paths.
                filename = Path(member_name).name

                parsed = preprocessor.parse_lobster_filename(Path(filename))

                # Non-LOBSTER members are consumed and ignored so that the
                # archive reader can continue correctly.
                if not parsed:
                    for _ in chunks:
                        pass
                    continue

                key = (
                    parsed["ticker"],
                    parsed["date"],
                    parsed["start"],
                    parsed["end"],
                    parsed["level"],
                )

                destination = temp_root / filename
                extract_entry(destination, chunks)

                pair = pending.setdefault(key, {})
                kind = parsed["kind"]

                if kind in pair:
                    raise RuntimeError(f"{ticker}: duplicate {kind} file for {key}")

                pair[kind] = destination

                if "message" not in pair or "orderbook" not in pair:
                    continue

                day = preprocessor.LobsterDay(
                    ticker=parsed["ticker"],
                    date=parsed["date"],
                    start_ms=parsed["start"],
                    end_ms=parsed["end"],
                    level=int(parsed["level"]),
                    message_path=pair["message"],
                    orderbook_path=pair["orderbook"],
                )

                minute, summary = preprocessor.aggregate_lobster_day(day)

                preprocessor.append_csv_gz(
                    minute,
                    panel_part,
                    write_header=not summaries,
                )

                summaries.append(summary)
                rows_written += len(minute)

                pair["message"].unlink()
                pair["orderbook"].unlink()
                del pending[key]

                print(
                    f"{ticker} {day.date}: {len(minute):,} minute rows; "
                    f"{summary['execution_count']:,} executions",
                    flush=True,
                )

        if pending:
            raise RuntimeError(
                f"{ticker}: {len(pending)} unpaired message/orderbook day(s) remain"
            )

        if not summaries:
            raise RuntimeError(f"{ticker}: no valid daily pairs found")

        pd.DataFrame(summaries).to_csv(daily_part, index=False)

        panel_part.replace(ticker_panel)
        daily_part.replace(ticker_daily)

        return {
            "ticker": ticker,
            "days": len(summaries),
            "rows": rows_written,
            "panel": str(ticker_panel),
            "daily_quality": str(ticker_daily),
            "skipped_existing": False,
        }

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def update_ticker_quality_from_file(panel_path: str) -> pd.DataFrame:
    """Compute ticker-level quality metrics from one compressed panel."""
    quality_source = pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=[
            "ticker",
            "model_ready_price",
            "bid_ask_spread_bps",
            "is_trading_halted",
        ],
    )

    ticker_quality = quality_source.groupby("ticker").agg(
        rows=("ticker", "size"),
        model_ready_rows=("model_ready_price", "sum"),
        median_bid_ask_spread_bps=("bid_ask_spread_bps", "median"),
        trading_halted_minutes=("is_trading_halted", "sum"),
    )

    ticker_quality["missing_model_ready_rows"] = (
        ticker_quality["rows"] - ticker_quality["model_ready_rows"]
    )

    return ticker_quality.reset_index()


def combine_outputs(results: list[dict], output_dir: Path) -> dict:
    """Combine per-ticker outputs into project-level files."""
    final_panel = output_dir / "lobster_minute_prices_model_ready.csv.gz"
    final_part = final_panel.with_suffix(final_panel.suffix + ".part")
    final_part.unlink(missing_ok=True)

    first = True

    for result in sorted(results, key=lambda item: item["ticker"]):
        for chunk in pd.read_csv(
            result["panel"],
            compression="gzip",
            chunksize=100_000,
        ):
            with gzip.open(final_part, "at", encoding="utf-8", newline="") as handle:
                chunk.to_csv(handle, index=False, header=first)
            first = False

    final_part.replace(final_panel)

    daily_frames = [pd.read_csv(result["daily_quality"]) for result in results]
    daily_quality = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["ticker", "date"]
    )

    daily_path = output_dir / "lobster_daily_quality.csv"
    daily_quality.to_csv(daily_path, index=False)

    ticker_quality_frames = [
        update_ticker_quality_from_file(result["panel"]) for result in results
    ]
    ticker_quality = pd.concat(ticker_quality_frames, ignore_index=True).sort_values(
        "ticker"
    )

    ticker_quality_path = output_dir / "lobster_ticker_quality.csv"
    ticker_quality.to_csv(ticker_quality_path, index=False)

    return {
        "output_path": str(final_panel),
        "daily_quality_path": str(daily_path),
        "ticker_quality_path": str(ticker_quality_path),
        "rows_written": int(sum(item["rows"] for item in results)),
        "days_processed": int(sum(item["days"] for item in results)),
        "tickers": sorted(item["ticker"] for item in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing annual LOBSTER .7z archives.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed outputs will be written.",
    )

    parser.add_argument(
        "--preprocessor",
        default=str(DEFAULT_PREPROCESSOR),
        help=(
            "Path to preprocess_lobster_intraday.py. Defaults to "
            "preprocessing/preprocess_lobster_intraday.py."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess tickers even if per-ticker outputs already exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    preprocessor_path = Path(args.preprocessor).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = load_project_preprocessor(preprocessor_path)

    archives = sorted(input_dir.glob("*.7z"))
    if not archives:
        raise SystemExit(f"No .7z archives found in {input_dir}")

    results: list[dict] = []
    progress_path = output_dir / "preprocessing_progress.json"

    for number, archive in enumerate(archives, start=1):
        print(f"\n[{number}/{len(archives)}] {archive.name}", flush=True)

        result = process_archive(
            archive=archive,
            output_dir=output_dir,
            preprocessor=preprocessor,
            force=args.force,
        )

        results.append(result)

        # Progress is written after each ticker so interrupted runs can be audited.
        progress_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = combine_outputs(results, output_dir)

    summary.update(
        {
            "input_archives": [str(path) for path in archives],
            "price_scaling": "LOBSTER integer prices divided by 10000",
            "model_price_definition": "top-of-book midpoint from level-1 bid/ask",
            "regular_hours": "09:30:00-16:00:00 America/New_York",
            "archive_handling": (
                "single-pass libarchive streaming; matched daily files are "
                "temporarily materialised before calling the project preprocessor"
            ),
            "preprocessor": str(preprocessor_path),
        }
    )

    summary_path = output_dir / "lobster_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
