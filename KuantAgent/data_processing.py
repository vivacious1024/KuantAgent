from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


# Core columns shared by the whole pipeline. Keeping this centralized
# avoids hard-coded field names in later analysis modules.
CORE_OHLC_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]
OPTIONAL_COLUMNS = ["Volume"]


@dataclass(frozen=True)
class OHLCWindowConfig:
    """Configuration for preparing a fixed-length OHLC analysis window."""

    window_size: int = 45
    keep_volume: bool = False
    datetime_format: str = "%Y-%m-%d %H:%M:%S"


COMMON_COLUMN_MAPPING = {
    "date": "Datetime",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "time": "Datetime",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "adjclose": "Close",
    "volume": "Volume",
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns into a flat one-level header."""

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for column in df.columns:
            parts = [str(part).strip() for part in column if str(part).strip()]
            flattened.append(parts[0] if parts else "")
        df = df.copy()
        df.columns = flattened
    return df


def _rename_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map provider-specific column names into the canonical OHLC schema."""

    rename_map = {}
    for column in df.columns:
        canonical = COMMON_COLUMN_MAPPING.get(str(column).strip().lower())
        if canonical:
            rename_map[column] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Fail fast when the upstream provider does not expose the required fields."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert price and volume fields into numeric types for downstream math."""

    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _filter_invalid_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with impossible OHLC relationships."""

    df = df.copy()
    price_columns = ["Open", "High", "Low", "Close"]

    positive_prices = (df[price_columns] > 0).all(axis=1)
    valid_high = df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)
    valid_low = df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)

    return df[positive_prices & valid_high & valid_low]


def normalize_ohlc_dataframe(
    raw_df: pd.DataFrame,
    keep_volume: bool = False,
) -> pd.DataFrame:
    """
    Standardize raw market data into a clean OHLC DataFrame.

    This step is intentionally independent from window slicing so that
    data ingestion and analysis-window construction stay decoupled.
    """

    if raw_df is None or raw_df.empty:
        raise ValueError("Input market data is empty.")

    df = raw_df.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = _flatten_columns(df)
    df = _rename_common_columns(df)

    # Some providers put the time column in the index rather than as a field.
    if "Datetime" not in df.columns and df.index.name:
        index_name = str(df.index.name).strip().lower()
        if index_name in {"date", "datetime", "timestamp", "time"}:
            df = df.reset_index()
            df = _rename_common_columns(df)

    # If the index still carries actual timestamps, promote it into a normal column.
    if "Datetime" not in df.columns and not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
        df = _rename_common_columns(df)

    required_columns: List[str] = CORE_OHLC_COLUMNS.copy()
    if keep_volume and "Volume" in df.columns:
        required_columns += ["Volume"]

    _ensure_required_columns(df, CORE_OHLC_COLUMNS)

    selected_columns = [column for column in CORE_OHLC_COLUMNS + OPTIONAL_COLUMNS if column in df.columns]
    df = df[selected_columns].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = _coerce_numeric_columns(df, ["Open", "High", "Low", "Close", "Volume"])

    df = df.dropna(subset=CORE_OHLC_COLUMNS)
    df = df.drop_duplicates(subset=["Datetime"], keep="last")
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = _filter_invalid_ohlc_rows(df).reset_index(drop=True)

    if keep_volume and "Volume" in df.columns:
        return df[CORE_OHLC_COLUMNS + ["Volume"]]
    return df[CORE_OHLC_COLUMNS]


def build_analysis_window(
    normalized_df: pd.DataFrame,
    config: OHLCWindowConfig | None = None,
) -> pd.DataFrame:
    """Slice the most recent fixed-length sample window for analysis."""

    config = config or OHLCWindowConfig()
    if normalized_df is None or normalized_df.empty:
        raise ValueError("Normalized market data is empty.")
    if len(normalized_df) < config.window_size:
        raise ValueError(
            f"Not enough rows for analysis window: need {config.window_size}, got {len(normalized_df)}"
        )
    return normalized_df.tail(config.window_size).reset_index(drop=True)


def dataframe_to_ohlc_dict(
    df: pd.DataFrame,
    config: OHLCWindowConfig | None = None,
) -> Dict[str, List]:
    """Convert the analysis window into the dict-of-lists format used downstream."""

    config = config or OHLCWindowConfig()
    ohlc_dict: Dict[str, List] = {}

    for column in CORE_OHLC_COLUMNS:
        if column == "Datetime":
            ohlc_dict[column] = df[column].dt.strftime(config.datetime_format).tolist()
        else:
            ohlc_dict[column] = df[column].astype(float).tolist()

    if config.keep_volume and "Volume" in df.columns:
        ohlc_dict["Volume"] = df["Volume"].fillna(0).astype(float).tolist()

    return ohlc_dict


def prepare_market_data(
    raw_df: pd.DataFrame,
    config: OHLCWindowConfig | None = None,
) -> Dict[str, object]:
    """Run the full preprocessing flow and return both tabular and dict outputs."""

    config = config or OHLCWindowConfig()
    normalized_df = normalize_ohlc_dataframe(raw_df, keep_volume=config.keep_volume)
    window_df = build_analysis_window(normalized_df, config=config)
    return {
        "normalized_df": normalized_df,
        "window_df": window_df,
        "ohlc_dict": dataframe_to_ohlc_dict(window_df, config=config),
    }
