from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _extract_last_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty stdout from integrated comparison runner.")

    decoder = json.JSONDecoder()
    last_payload: Dict[str, Any] | None = None
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "output_dir" in payload:
            last_payload = payload
    if last_payload is None:
        raise ValueError(f"Could not parse output JSON from runner stdout:\n{cleaned[:2000]}")
    return last_payload


def _run_one(
    *,
    benchmark_dir: Path,
    timeframe: str,
    index: int,
    window_size: int,
    future_horizon: int,
    neutral_threshold_pct: float,
    system_timeout_sec: int,
) -> Dict[str, Any]:
    script_path = Path(__file__).resolve().parent / "run_integrated_comparison.py"
    command = [
        sys.executable,
        str(script_path),
        "--benchmark-dir",
        str(benchmark_dir),
        "--timeframe",
        timeframe,
        "--window-size",
        str(window_size),
        "--future-horizon",
        str(future_horizon),
        "--neutral-threshold-pct",
        str(neutral_threshold_pct),
        "--run-kuant-full",
        "--start-index",
        str(index),
        "--end-index",
        str(index),
        "--limit",
        "1",
        "--system-timeout-sec",
        str(system_timeout_sec),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = _extract_last_json(completed.stdout)
    summary = payload.get("summary", {}) or {}
    kuant = ((summary.get("kuant_full") or {}).get("accuracy_report") or {})
    pure = ((summary.get("pure_algo") or {}).get("accuracy_report") or {})
    kuant_detail = (kuant.get("details") or [{}])[0]
    parsed = kuant_detail.get("parsed_decision", {}) or {}
    return {
        "index": index,
        "output_dir": payload.get("output_dir", ""),
        "sample_file": kuant_detail.get("sample_file", ""),
        "kuant_predicted": kuant_detail.get("predicted", ""),
        "true_direction": kuant_detail.get("true_direction", ""),
        "kuant_correct": kuant_detail.get("correct", 0),
        "kuant_confidence": kuant_detail.get("confidence", 0.0),
        "base_source": parsed.get("base_source", ""),
        "arb_winner": ((parsed.get("expert_arbitration") or {}).get("winner", "")),
        "arb_reason": ((parsed.get("expert_arbitration") or {}).get("reason", "")),
        "semantic": parsed.get("structure_semantic_label", ""),
        "path_bias": parsed.get("three_bar_majority_bias", ""),
        "path_consistency": parsed.get("three_bar_path_consistency", 0.0),
        "path_score": parsed.get("three_bar_path_score", 0.0),
        "authority": parsed.get("decision_authority_regime", ""),
        "execution_advice": parsed.get("execution_advice", ""),
        "pure_predicted": ((pure.get("details") or [{}])[0]).get("predicted", ""),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KuantAgent single-sample experiments in parallel.")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.15)
    parser.add_argument("--system-timeout-sec", type=int, default=2400)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(
                _run_one,
                benchmark_dir=benchmark_dir,
                timeframe=args.timeframe,
                index=index,
                window_size=args.window_size,
                future_horizon=args.future_horizon,
                neutral_threshold_pct=args.neutral_threshold_pct,
                system_timeout_sec=args.system_timeout_sec,
            )
            for index in args.indices
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["index"])
    output_root = Path(__file__).resolve().parent / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = output_root / f"parallel_kuant_single_samples_{run_id}.json"
    report_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
