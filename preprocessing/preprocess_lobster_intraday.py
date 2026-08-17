
"""
Preprocess LOBSTER message/orderbook files into minute-level data.

The output will be: one row per ticker-minute
with top-of-book bid/ask, midquote OHLC, execution volume/counts, and quality
flags. 
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOBSTER_RE = re.compile(
    r"^(?P<ticker>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<kind>message|orderbook)_(?P<level>\d+)\.csv(?:\.gz)?$"
)


MESSAGE_COLUMNS = ["seconds", "type", "order_id", "size", "price", "direction"]
ORDERBOOK_COLUMNS_LEVEL_1 = ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]

MARKET_OPEN = "09:30:00"
MARKET_CLOSE = "16:00:00"
MARKET_OPEN_SECONDS = 9 * 60 * 60 + 30 * 60
MARKET_CLOSE_SECONDS = 16 * 60 * 60
MARKET_CLOSE_BAR_END_SECONDS = MARKET_CLOSE_SECONDS


@dataclass(frozen=True)
class LobsterDay:
    ticker: str
    date: str
    start_ms: str
    end_ms: str
    level: int
    message_path: Path
    orderbook_path: Path


def parse_lobster_filename(path: Path) -> dict[str, str] | None:
    match = LOBSTER_RE.match(path.name)
    if not match:
        return None
    return match.groupdict()


def discover_lobster_days(input_paths: Iterable[Path]) -> list[LobsterDay]:
    """Find matched daily message/orderbook file pairs.

    Files are matched by ticker, date, trading window and book level. Days with
    a missing message or orderbook file are skipped.
    """
    messages: dict[tuple[str, str, str, str, str], Path] = {}
    orderbooks: dict[tuple[str, str, str, str, str], Path] = {}

    for input_path in input_paths:
        paths = (
            list(input_path.glob("*.csv")) + list(input_path.glob("*.csv.gz"))
            if input_path.is_dir()
            else [input_path]
        )
        for path in paths:
            parsed = parse_lobster_filename(path)
            if not parsed:
                continue
            key = (
                parsed["ticker"],
                parsed["date"],
                parsed["start"],
                parsed["end"],
                parsed["level"],
            )
            if parsed["kind"] == "message":
                messages[key] = path
            else:
                orderbooks[key] = path


    days: list[LobsterDay] = []
    for key, message_path in sorted(messages.items(), key=lambda item: item[0]):
        orderbook_path = orderbooks.get(key)
        if orderbook_path is None:
            continue
        ticker, date, start_ms, end_ms, level = key
        days.append(
            LobsterDay(
                ticker=ticker,
                date=date,
                start_ms=start_ms,
                end_ms=end_ms,
                level=int(level),
                message_path=message_path,
                orderbook_path=orderbook_path,
            )
        )
    return days


def price_to_dollars(series: pd.Series) -> pd.Series:
    """Convert LOBSTER integer prices to dollar prices.
    LOBSTER stores prices multiplied by 10,000. Dummy quote values are treated
    as missing before scaling.
    """
    raw = pd.to_numeric(series, errors="coerce")
    raw = raw.mask(raw.isin([9999999999, -9999999999]))
    return raw / 10000.0


def read_lobster_day(day: LobsterDay) -> pd.DataFrame:
    """Read one LOBSTER trading day and construct event-level top-of-book data.

    Message and orderbook files are aligned row-by-row, as required by the
    LOBSTER data format. Level 1 bid/ask quotes are used to construct the
    midpoint and bid-ask spread.
    """
    message = pd.read_csv(
        day.message_path,
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=MESSAGE_COLUMNS,
    )
    orderbook = pd.read_csv(
        day.orderbook_path,
        header=None,
        usecols=[0, 1, 2, 3],
        names=ORDERBOOK_COLUMNS_LEVEL_1,
    )
    if len(message) != len(orderbook):
        raise ValueError(
            f"Message/orderbook row mismatch for {day.ticker} {day.date}: "
            f"{len(message)} vs {len(orderbook)}"
        )

    frame = pd.concat([message, orderbook], axis=1)

    frame["ticker"] = day.ticker
    frame["level"] = day.level
    frame["source_message_file"] = day.message_path.name
    frame["source_orderbook_file"] = day.orderbook_path.name
    frame["seconds"] = pd.to_numeric(frame["seconds"], errors="coerce")
    frame["minute_number"] = np.floor(frame["seconds"] / 60).astype("int64")
    frame["ask_price_1"] = price_to_dollars(frame["ask_price_1"])
    frame["bid_price_1"] = price_to_dollars(frame["bid_price_1"])
    frame["message_price"] = price_to_dollars(frame["price"])
    frame["mid_price"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    frame["bid_ask_spread"] = frame["ask_price_1"] - frame["bid_price_1"]
    frame["bid_ask_spread_bps"] = 10000.0 * frame["bid_ask_spread"] / frame["mid_price"]
    frame["valid_quote"] = (
        frame["ask_price_1"].notna()
        & frame["bid_price_1"].notna()
        & (frame["ask_price_1"] >= frame["bid_price_1"]))
    frame["is_visible_execution"] = frame["type"].eq(4)
    frame["is_hidden_execution"] = frame["type"].eq(5)
    frame["is_execution"] = frame["is_visible_execution"] | frame["is_hidden_execution"]
    frame["is_trading_halt_message"] = frame["type"].eq(7)
    frame["buyer_initiated_execution"] = frame["is_execution"] & frame["direction"].eq(-1)
    frame["seller_initiated_execution"] = frame["is_execution"] & frame["direction"].eq(1)
    return frame


def ohlc_by_minute(frame: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:

    grouped = (
        frame.dropna(subset=[value_col])
        .groupby("minute_number")[value_col]
        .agg(["first", "max", "min", "last"])
        .reset_index()
    )
    return grouped.rename(
        columns={
            "first": f"{prefix}_open",
            "max": f"{prefix}_high",
            "min": f"{prefix}_low",
            "last": f"{prefix}_close",
        }
    )


def fill_empty_minute_ohlc(minute: pd.DataFrame, prefixes: Iterable[str]) -> None:
    """Forward-fill empty quote minutes to maintain a regular minute grid.

    If no quote update occurs in a minute, the previous close is carried forward
    and used as that minute's open/high/low/close. This creates a complete
    390-row trading day panel for model estimation.
    """
    for prefix in prefixes:
        close_col = f"{prefix}_close"
        actual_minute = minute[close_col].notna()
        filled_close = minute[close_col].ffill()
        minute[close_col] = filled_close
        for part in ["open", "high", "low"]:
            col = f"{prefix}_{part}"
            minute[col] = minute[col].where(actual_minute, filled_close)


def halted_minute_flags(events: pd.DataFrame, minute_numbers: pd.Series) -> pd.Series:
    """Mark minutes affected by LOBSTER trading halt messages.

    Halt intervals are inferred from type-7 messages and excluded from the
    model-ready sample.
    """
    halt_events = events[
        events["is_trading_halt_message"]
        & (events["seconds"] >= MARKET_OPEN_SECONDS)
        & (events["seconds"] < MARKET_CLOSE_SECONDS)
    ][["seconds", "price"]].copy()
    if halt_events.empty:
        return pd.Series(False, index=minute_numbers.index)

    halt_events["halt_status"] = pd.to_numeric(halt_events["price"], errors="coerce")
    halt_events = halt_events.sort_values("seconds")

    intervals: list[tuple[float, float]] = []
    halt_start: float | None = None
    for row in halt_events.itertuples(index=False):
        seconds = float(row.seconds)
        if row.halt_status == -1:
            halt_start = seconds
        elif row.halt_status == 1 and halt_start is not None:
            halt_end = min(seconds, float(MARKET_CLOSE_SECONDS))
            if halt_end > halt_start:
                intervals.append((halt_start, halt_end))
            halt_start = None

    if halt_start is not None:
        intervals.append((halt_start, float(MARKET_CLOSE_SECONDS)))

    if not intervals:
        return pd.Series(False, index=minute_numbers.index)

    minute_start_seconds = minute_numbers * 60
    minute_end_seconds = minute_start_seconds + 60
    halted = pd.Series(False, index=minute_numbers.index)
    for halt_start, halt_end in intervals:
        halted |= (minute_start_seconds < halt_end) & (minute_end_seconds > halt_start)
    return halted


def aggregate_lobster_day(day: LobsterDay) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate one LOBSTER day into minute-level model inputs.

    The model price is the top-of-book midpoint. Bid and ask prices are retained
    separately so that backtests can apply realistic transaction costs.
    """
    events = read_lobster_day(day)
    regular_hours = (
        (events["seconds"] >= MARKET_OPEN_SECONDS)
        & (events["seconds"] < MARKET_CLOSE_BAR_END_SECONDS)
    )
    regular = events[
        regular_hours
        & events["valid_quote"]
    ].copy()

    minute_index = pd.date_range(
        pd.Timestamp(f"{day.date} {MARKET_OPEN}").tz_localize("America/New_York"),
        pd.Timestamp(f"{day.date} {MARKET_CLOSE}").tz_localize("America/New_York"),
        freq="min",
        inclusive="left",
    )
    base = pd.DataFrame({"bar_minute_ny_dt": minute_index})
    base["bar_minute_utc_dt"] = base["bar_minute_ny_dt"].dt.tz_convert("UTC")
    base["timestamp_ny"] = base["bar_minute_ny_dt"].astype(str)
    base["timestamp_utc"] = base["bar_minute_utc_dt"].astype(str)
    base["bar_minute_ny"] = base["timestamp_ny"]
    base["bar_minute_utc"] = base["timestamp_utc"]
    base["trade_date"] = base["bar_minute_ny_dt"].dt.date.astype(str)
    base["market_time"] = base["bar_minute_ny_dt"].dt.strftime("%H:%M:%S")
    base["minute_number"] = (
        base["bar_minute_ny_dt"].dt.hour * 60
        + base["bar_minute_ny_dt"].dt.minute
    )

    if regular.empty:
        minute = base.copy()
        for col in [
            "model_price_open",
            "model_price_high",
            "model_price_low",
            "model_price_close",
            "bid_close",
            "ask_close",
            "mid_close",
            "bid_ask_spread",
            "bid_ask_spread_bps",
        ]:
            minute[col] = np.nan
    else:
        mid_ohlc = ohlc_by_minute(regular, "mid_price", "model_price")
        bid_ohlc = ohlc_by_minute(regular, "bid_price_1", "bid")
        ask_ohlc = ohlc_by_minute(regular, "ask_price_1", "ask")
        closes = regular.groupby("minute_number").agg(
            bid_size_close=("bid_size_1", "last"),
            ask_size_close=("ask_size_1", "last"),
            event_count=("seconds", "size"),
            source_message_file=("source_message_file", "last"),
            source_orderbook_file=("source_orderbook_file", "last"),
        )
        minute = (
            base.merge(mid_ohlc, on="minute_number", how="left")
            .merge(bid_ohlc, on="minute_number", how="left")
            .merge(ask_ohlc, on="minute_number", how="left")
            .merge(closes.reset_index(), on="minute_number", how="left")
        )

        fill_empty_minute_ohlc(minute, ["model_price", "bid", "ask"])
        minute[["bid_size_close", "ask_size_close"]] = minute[
            ["bid_size_close", "ask_size_close"]
        ].ffill()
        minute["mid_close"] = (minute["bid_close"] + minute["ask_close"]) / 2.0
        minute["bid_ask_spread"] = minute["ask_close"] - minute["bid_close"]
        minute["bid_ask_spread_bps"] = 10000.0 * minute["bid_ask_spread"] / minute["mid_close"]

    executions = events[
        regular_hours
        & events["is_execution"]
    ].copy()
    if executions.empty:
        exec_minute = pd.DataFrame({"minute_number": []})
    else:
        execution_ohlc = ohlc_by_minute(executions, "message_price", "execution_price")
        executions["buyer_initiated_size"] = np.where(
            executions["buyer_initiated_execution"], executions["size"], 0
        )
        executions["seller_initiated_size"] = np.where(
            executions["seller_initiated_execution"], executions["size"], 0
        )
        execution_counts = executions.groupby("minute_number").agg(
            execution_volume=("size", "sum"),
            execution_count=("size", "size"),
            visible_execution_count=("is_visible_execution", "sum"),
            hidden_execution_count=("is_hidden_execution", "sum"),
            buyer_initiated_volume=("buyer_initiated_size", "sum"),
            seller_initiated_volume=("seller_initiated_size", "sum"),
        )
        exec_minute = execution_ohlc.merge(execution_counts.reset_index(), on="minute_number", how="outer")

    minute = minute.merge(exec_minute, on="minute_number", how="left")
    count_cols = [
        "event_count",
        "execution_volume",
        "execution_count",
        "visible_execution_count",
        "hidden_execution_count",
        "buyer_initiated_volume",
        "seller_initiated_volume",
    ]
    for col in count_cols:
        if col not in minute:
            minute[col] = 0
        minute[col] = minute[col].fillna(0)

    halt_by_minute = (
        events[
            regular_hours
            & events["is_trading_halt_message"]
        ]
        .groupby("minute_number")
        .size()
        .rename("trading_halt_message_count")
        .reset_index()
    )
    minute = minute.merge(halt_by_minute, on="minute_number", how="left")
    minute["trading_halt_message_count"] = minute["trading_halt_message_count"].fillna(0)
    minute["is_trading_halted"] = halted_minute_flags(events, minute["minute_number"])

    minute["ticker"] = day.ticker
    minute["level"] = day.level
    minute["source"] = "LOBSTER"
    minute["source_message_file"] = day.message_path.name
    minute["source_orderbook_file"] = day.orderbook_path.name
    # Boolean flag: True if this minute has usable bid/ask/midquote data for modelling.
    minute["model_ready_price"] = (
        minute["model_price_close"].notna()
        & minute["bid_close"].notna()
        & minute["ask_close"].notna()
        & (minute["ask_close"] >= minute["bid_close"])
        & ~minute["is_trading_halted"]
    )

    output_cols = [
        "timestamp_utc",
        "timestamp_ny",
        "trade_date",
        "market_time",
        "ticker",
        "source",
        "level",
        "model_price_open",
        "model_price_high",
        "model_price_low",
        "model_price_close",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "mid_close",
        "bid_ask_spread",
        "bid_ask_spread_bps",
        "bid_size_close",
        "ask_size_close",
        "execution_price_open",
        "execution_price_high",
        "execution_price_low",
        "execution_price_close",
        "execution_volume",
        "execution_count",
        "visible_execution_count",
        "hidden_execution_count",
        "buyer_initiated_volume",
        "seller_initiated_volume",
        "event_count",
        "trading_halt_message_count",
        "is_trading_halted",
        "model_ready_price",
        "source_message_file",
        "source_orderbook_file",
    ]
    for col in output_cols:
        if col not in minute:
            minute[col] = np.nan
    minute = minute[output_cols]

    spread_values = minute["bid_ask_spread_bps"].dropna()
    median_spread_bps = float(spread_values.median()) if not spread_values.empty else None
    summary = {
        "ticker": day.ticker,
        "date": day.date,
        "level": day.level,
        "raw_events": int(len(events)),
        "regular_valid_quote_events": int(len(regular)),
        "minute_rows": int(len(minute)),
        "model_ready_rows": int(minute["model_ready_price"].sum()),
        "execution_count": int(minute["execution_count"].sum()),
        "execution_volume": int(minute["execution_volume"].sum()),
        "trading_halt_message_count": int(minute["trading_halt_message_count"].sum()),
        "trading_halted_minutes": int(minute["is_trading_halted"].sum()),
        "median_spread_bps": median_spread_bps,
    }
    return minute, summary


def append_csv_gz(df: pd.DataFrame, path: Path, write_header: bool) -> None:
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False, header=write_header)


def preprocess_lobster(
    input_paths: list[Path],
    output_dir: Path,
    max_days: int | None = None,
) -> dict[str, object]:
    """Run the full preprocessing pipeline and write model-ready outputs.

    This function is importable from other runners, including archive-streaming
    scripts, and can also be called through the command-line interface below.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    days = discover_lobster_days(input_paths)
    if max_days is not None:
        days = days[:max_days]
    if not days:
        raise FileNotFoundError("No matched LOBSTER message/orderbook day pairs found.")

    output_path = output_dir / "lobster_minute_prices_model_ready.csv.gz"
    if output_path.exists():
        output_path.unlink()

    day_summaries = []
    total_rows = 0
    for idx, day in enumerate(days, start=1):
        minute, summary = aggregate_lobster_day(day)
        append_csv_gz(minute, output_path, write_header=idx == 1)
        day_summaries.append(summary)
        total_rows += len(minute)
        print(
            f"{idx:04d}/{len(days)} {day.ticker} {day.date}: "
            f"{len(minute):,} minute rows, {summary['execution_count']:,} executions"
        )

    day_summary_df = pd.DataFrame(day_summaries)
    day_summary_path = output_dir / "lobster_daily_quality.csv"
    day_summary_df.to_csv(day_summary_path, index=False)

    panel = pd.read_csv(
        output_path,
        compression="gzip",
        usecols=["ticker", "model_ready_price", "bid_ask_spread_bps", "is_trading_halted"],
    )
    ticker_quality = panel.groupby("ticker").agg(
        rows=("ticker", "size"),
        model_ready_rows=("model_ready_price", "sum"),
        median_bid_ask_spread_bps=("bid_ask_spread_bps", "median"),
        trading_halted_minutes=("is_trading_halted", "sum"),
    )
    ticker_quality["missing_model_ready_rows"] = ticker_quality["rows"] - ticker_quality["model_ready_rows"]
    ticker_quality_path = output_dir / "lobster_ticker_quality.csv"
    ticker_quality.reset_index().to_csv(ticker_quality_path, index=False)

    summary = {
        "inputs": [str(path) for path in input_paths],
        "days_processed": len(days),
        "tickers": sorted({day.ticker for day in days}),
        "output_path": str(output_path),
        "rows_written": total_rows,
        "daily_quality_path": str(day_summary_path),
        "ticker_quality_path": str(ticker_quality_path),
        "price_scaling": "LOBSTER integer prices divided by 10000",
        "model_price_definition": "top-of-book midpoint from level-1 bid/ask",
        "regular_hours": f"{MARKET_OPEN}-{MARKET_CLOSE} America/New_York",
    }
    summary_path = output_dir / "lobster_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="LOBSTER ticker folders or files")
    parser.add_argument("--output-dir", default="data/processed_lobster", help="Output directory")
    parser.add_argument("--max-days", type=int, default=None, help="Debug option: process only the first N days")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_lobster(
        input_paths=[Path(path).expanduser() for path in args.inputs],
        output_dir=Path(args.output_dir),
        max_days=args.max_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
