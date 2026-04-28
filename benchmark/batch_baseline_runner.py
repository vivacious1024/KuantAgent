from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
KUANT_DIR = PROJECT_ROOT / "KuantAgent"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(KUANT_DIR) not in sys.path:
    sys.path.insert(0, str(KUANT_DIR))

from baseline_runner import run_algorithmic_baseline


def infer_asset_from_filename(csv_path: Path) -> str:
    """Infer asset symbol from a benchmark filename like BTC_1h_1.csv."""

    return csv_path.stem.split("_")[0]


def _infer_asset_from_filename(csv_path: str | Path) -> str:
    """Backward-compatible helper for older benchmark scripts."""

    return infer_asset_from_filename(Path(csv_path))


def benchmark_sort_key(csv_path: Path) -> tuple[str, str, int, str]:
    """
    Sort benchmark files by numeric sample index instead of lexicographic filename.

    Example:
    BTC_1h_2.csv should come before BTC_1h_10.csv.
    """

    stem_parts = csv_path.stem.split("_")
    asset = stem_parts[0] if stem_parts else ""
    timeframe = stem_parts[1] if len(stem_parts) > 1 else ""
    try:
        sample_index = int(stem_parts[-1])
    except Exception:
        sample_index = 10**9
    return asset, timeframe, sample_index, csv_path.name


def run_directory_baseline(
    benchmark_dir: str | Path,
    timeframe: str,
    window_size: int = 45,
    future_horizon: int = 3,
    neutral_threshold_pct: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Run the deterministic baseline on every CSV file under a benchmark directory.

    Why this matters:
    One-sample testing is useful for debugging, but quant research needs batch
    statistics. This runner is the bridge from manual inspection to automated
    evaluation.
    """

    benchmark_dir = Path(benchmark_dir)
    csv_files = sorted(benchmark_dir.rglob("*.csv"), key=benchmark_sort_key)
    results: List[Dict[str, Any]] = []

    for csv_file in csv_files:
        asset = infer_asset_from_filename(csv_file)
        result = run_algorithmic_baseline(
            csv_path=csv_file,
            asset=asset,
            timeframe=timeframe,
            window_size=window_size,
            future_horizon=future_horizon,
            neutral_threshold_pct=neutral_threshold_pct,
        )
        results.append(result)

    return results


def save_batch_results(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Persist batch baseline outputs as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
