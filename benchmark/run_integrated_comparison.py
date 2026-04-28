from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from batch_baseline_runner import benchmark_sort_key, infer_asset_from_filename
from baseline_runner import run_algorithmic_baseline


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Subprocess stdout is empty.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    last_payload: Dict[str, Any] | None = None
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "sample_file" in payload:
            last_payload = payload

    if last_payload is None:
        raise ValueError(f"Could not find a valid result JSON object in subprocess stdout:\n{cleaned[:1000]}")

    return last_payload


def _forward_pipe_lines(pipe: Any, collector: List[str], prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            collector.append(line)
            text = line.rstrip()
            if text:
                print(f"{prefix}{text}", file=sys.stderr, flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_full_system_subprocess(
    helper_script: Path,
    project_root: Path,
    csv_path: Path,
    asset: str,
    timeframe: str,
    window_size: int,
    future_horizon: int,
    neutral_threshold_pct: float,
    timeout_sec: int,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(helper_script),
        "--project-root",
        str(project_root),
        "--csv-path",
        str(csv_path),
        "--asset",
        asset,
        "--timeframe",
        timeframe,
        "--window-size",
        str(window_size),
        "--future-horizon",
        str(future_horizon),
        "--neutral-threshold-pct",
        str(neutral_threshold_pct),
    ]
    command_text = subprocess.list2cmdline(command)
    print(
        f"[runner] Launching {project_root.name} helper script for {csv_path.name}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[runner] Command: {command_text}", file=sys.stderr, flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    if stdout_pipe is None or stderr_pipe is None:
        process.kill()
        raise RuntimeError("Failed to create subprocess pipes for helper script.")

    stderr_lines: List[str] = []
    stderr_prefix = f"[{project_root.name}:{csv_path.name}] "
    stderr_thread = threading.Thread(
        target=_forward_pipe_lines,
        args=(stderr_pipe, stderr_lines, stderr_prefix),
        daemon=True,
    )
    stderr_thread.start()

    try:
        stdout_text = stdout_pipe.read()
        returncode = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stderr_thread.join(timeout=2)
        stderr_text = "".join(stderr_lines).strip()
        raise RuntimeError(
            "Full-system subprocess timed out.\n"
            f"Command: {command_text}\n"
            f"Timeout (sec): {timeout_sec}\n"
            f"Details:\n{stderr_text or 'No subprocess output captured before timeout.'}"
        ) from exc
    finally:
        try:
            stdout_pipe.close()
        except Exception:
            pass

    stderr_thread.join(timeout=2)
    stderr_text = "".join(stderr_lines).strip()

    if returncode != 0:
        details = stderr_text or stdout_text or "No subprocess output captured."
        raise RuntimeError(
            "Full-system subprocess failed.\n"
            f"Command: {command_text}\n"
            f"Details:\n{details}"
        )
    try:
        return _extract_json_payload(stdout_text)
    except Exception as exc:
        raise RuntimeError(
            "Full-system subprocess completed but did not return valid JSON.\n"
            f"Command: {command_text}\n"
            f"Parse error: {exc}\n"
            f"STDOUT:\n{stdout_text[:2000] or '<empty>'}\n"
            f"STDERR:\n{stderr_text[:2000] or '<empty>'}"
        ) from exc


def _to_eval_record(result: Dict[str, Any]) -> Dict[str, Any]:
    predicted_value = result.get("predicted", result.get("dominant_side", ""))
    predicted = str(predicted_value).upper()
    parsed_decision = result.get("parsed_decision", {}) or {}
    execution_advice = str(parsed_decision.get("execution_advice", "")).lower()
    hard_case_score = float(parsed_decision.get("hard_case_score", 0.0) or 0.0)
    is_neutral_prediction = int(predicted == "NEUTRAL" or execution_advice == "skip")
    return {
        "asset": result["asset"],
        "timeframe": result["timeframe"],
        "sample_file": result["sample_file"],
        "predicted": predicted,
        "true_direction": result["true_direction"],
        "correct": int(result["correct"]),
        "confidence": float(result["confidence"] or 0.0),
        "future_return_pct": float(result["future_return_pct"]),
        "horizon_correct_count": int(result.get("horizon_correct_count", int(result["correct"])) or 0),
        "horizon_total_count": int(result.get("horizon_total_count", 1) or 1),
        "horizon_step_accuracy": float(result.get("horizon_step_accuracy", float(result["correct"])) or 0.0),
        "future_step_directions": result.get("future_step_directions", []),
        "is_neutral_move": int(result["is_neutral_move"]),
        "is_neutral_prediction": is_neutral_prediction,
        "raw_predicted": result.get("raw_predicted", ""),
        "consensus_level": result.get("consensus_level", ""),
        "risk_level": result.get("risk_level", ""),
        "decision_route": parsed_decision.get("decision_route", ""),
        "route_reason": parsed_decision.get("route_reason", ""),
        "ai_review_skipped": int(bool(parsed_decision.get("ai_review_skipped", False))),
        "execution_grade": parsed_decision.get("execution_grade", ""),
        "hard_case_score": hard_case_score,
        "is_hard_case": int(hard_case_score >= 0.3),
        "is_system_failure": int(bool(parsed_decision.get("system_error")) or parsed_decision.get("decision_route") == "system_failure"),
        "raw_decision": result.get("raw_decision", ""),
        "parsed_decision": parsed_decision,
    }


def _failure_eval_record(
    *,
    asset: str,
    timeframe: str,
    sample_file: str,
    true_direction: str,
    future_return_pct: float,
    is_neutral_move: int,
    horizon_total_count: int,
    future_step_directions: List[str],
    error: str,
) -> Dict[str, Any]:
    """Create a non-actionable placeholder record so one failing sample does not abort the whole benchmark."""

    return {
        "asset": asset,
        "timeframe": timeframe,
        "sample_file": sample_file,
        "predicted": "NEUTRAL",
        "true_direction": true_direction,
        "correct": 0,
        "confidence": 0.0,
        "future_return_pct": future_return_pct,
        "horizon_correct_count": 0,
        "horizon_total_count": horizon_total_count,
        "horizon_step_accuracy": 0.0,
        "future_step_directions": future_step_directions,
        "is_neutral_move": is_neutral_move,
        "is_neutral_prediction": 1,
        "raw_predicted": "",
        "consensus_level": "",
        "risk_level": "",
        "decision_route": "system_failure",
        "route_reason": error,
        "ai_review_skipped": 0,
        "execution_grade": "",
        "hard_case_score": 0.0,
        "is_hard_case": 0,
        "is_system_failure": 1,
        "raw_decision": "",
        "parsed_decision": {"system_error": error},
    }


def _summarize_system(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "accuracy_report": {
                "num_samples": 0,
                "accuracy": None,
                "avg_confidence": None,
                "num_neutral_predictions": 0,
                "actionable_num_samples": 0,
                "actionable_coverage": 0.0,
                "actionable_accuracy": None,
                "num_neutral_moves": 0,
                "filtered_num_samples": 0,
                "filtered_accuracy_ex_neutral": None,
                "filtered_actionable_num_samples": 0,
                "filtered_actionable_accuracy": None,
                "asset_accuracy": {},
                "details": [],
            }
        }

    df = pd.DataFrame(records)
    accuracy = round(float(df["correct"].mean()), 4)
    horizon_total = int(df["horizon_total_count"].sum()) if "horizon_total_count" in df.columns else int(len(df))
    horizon_correct = int(df["horizon_correct_count"].sum()) if "horizon_correct_count" in df.columns else int(df["correct"].sum())
    horizon_step_accuracy = None if horizon_total == 0 else round(float(horizon_correct / horizon_total), 4)
    avg_confidence = round(float(df["confidence"].mean()), 4)
    actionable_df = df[df["is_neutral_prediction"] == 0]
    strong_move_df = df[df["is_neutral_move"] == 0]
    strong_move_actionable_df = strong_move_df[strong_move_df["is_neutral_prediction"] == 0]
    hard_case_df = df[df["is_hard_case"] == 1]
    hard_case_actionable_df = hard_case_df[hard_case_df["is_neutral_prediction"] == 0]
    successful_df = df[df["is_system_failure"] == 0]

    asset_accuracy = df.groupby("asset")["correct"].mean().round(4).to_dict()

    return {
        "accuracy_report": {
            "num_samples": int(len(df)),
            "accuracy": accuracy,
            "horizon_step_accuracy": horizon_step_accuracy,
            "horizon_correct_count": horizon_correct,
            "horizon_total_count": horizon_total,
            "avg_confidence": avg_confidence,
            "num_neutral_predictions": int(df["is_neutral_prediction"].sum()),
            "actionable_num_samples": int(len(actionable_df)),
            "actionable_coverage": round(float(len(actionable_df) / len(df)), 4),
            "actionable_accuracy": None if actionable_df.empty else round(float(actionable_df["correct"].mean()), 4),
            "num_neutral_moves": int(df["is_neutral_move"].sum()),
            "filtered_num_samples": int(len(strong_move_df)),
            "filtered_accuracy_ex_neutral": None if strong_move_df.empty else round(float(strong_move_df["correct"].mean()), 4),
            "filtered_actionable_num_samples": int(len(strong_move_actionable_df)),
            "filtered_actionable_accuracy": None
            if strong_move_actionable_df.empty
            else round(float(strong_move_actionable_df["correct"].mean()), 4),
            "hard_case_num_samples": int(len(hard_case_df)),
            "hard_case_accuracy": None if hard_case_df.empty else round(float(hard_case_df["correct"].mean()), 4),
            "hard_case_actionable_num_samples": int(len(hard_case_actionable_df)),
            "hard_case_actionable_accuracy": None
            if hard_case_actionable_df.empty
            else round(float(hard_case_actionable_df["correct"].mean()), 4),
            "algorithm_only_num_samples": int((df["decision_route"] == "algorithm_only").sum()),
            "algorithm_only_coverage": round(float((df["decision_route"] == "algorithm_only").mean()), 4),
            "system_failure_count": int(df["is_system_failure"].sum()),
            "successful_num_samples": int(len(successful_df)),
            "successful_coverage": round(float(len(successful_df) / len(df)), 4),
            "asset_accuracy": asset_accuracy,
            "details": df.to_dict(orient="records"),
        }
    }


def _sample_index_from_path(sample_file: str) -> int | None:
    stem = Path(sample_file).stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return None


def _write_system_outputs(system_dir: Path, raw_results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "results.json").write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(raw_results).to_csv(system_dir / "results.csv", index=False, encoding="utf-8-sig")
    (system_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_output(
    output_dir: Path,
    pure_results: List[Dict[str, Any]],
    kuant_results: List[Dict[str, Any]],
    quant_results: List[Dict[str, Any]],
) -> None:
    pure_summary = _summarize_system(pure_results)
    kuant_summary = _summarize_system(kuant_results) if kuant_results else None
    quant_summary = _summarize_system(quant_results) if quant_results else None

    _write_system_outputs(output_dir / "pure_algo", pure_results, pure_summary)
    if kuant_summary is not None:
        _write_system_outputs(output_dir / "kuant_full", kuant_results, kuant_summary)
    if quant_summary is not None:
        _write_system_outputs(output_dir / "quant_full", quant_results, quant_summary)

    combined_summary = {
        "pure_algo": pure_summary,
        "kuant_full": kuant_summary,
        "quant_full": quant_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_report(
    benchmark_dir: Path,
    output_dir: Path,
    pure_summary: Dict[str, Any],
    kuant_summary: Dict[str, Any] | None,
    quant_summary: Dict[str, Any] | None,
) -> str:
    lines: List[str] = []
    lines.append("# Comprehensive Comparison Report")
    lines.append("")
    lines.append(f"- Benchmark directory: `{benchmark_dir}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("")

    def add_system_block(name: str, summary: Dict[str, Any] | None) -> None:
        lines.append(f"## {name}")
        if summary is None:
            lines.append("- Not run")
            lines.append("")
            return
        report = summary["accuracy_report"]
        lines.append(f"- Num samples: `{report['num_samples']}`")
        lines.append(f"- Accuracy: `{report['accuracy']}`")
        lines.append(f"- Horizon step accuracy: `{report.get('horizon_step_accuracy')}`")
        lines.append(f"- Avg confidence: `{report['avg_confidence']}`")
        lines.append(f"- Actionable accuracy: `{report['actionable_accuracy']}`")
        lines.append(f"- Actionable coverage: `{report['actionable_coverage']}`")
        lines.append(f"- Filtered accuracy ex neutral: `{report['filtered_accuracy_ex_neutral']}`")
        lines.append(f"- Hard-case accuracy: `{report['hard_case_accuracy']}`")
        lines.append(f"- Hard-case actionable accuracy: `{report['hard_case_actionable_accuracy']}`")
        lines.append(f"- Algorithm-only coverage: `{report['algorithm_only_coverage']}`")
        lines.append(f"- System failure count: `{report['system_failure_count']}`")
        lines.append(f"- Successful coverage: `{report['successful_coverage']}`")
        lines.append("")

    add_system_block("Pure Algorithm Baseline", pure_summary)
    add_system_block("KuantAgent Full System", kuant_summary)
    add_system_block("QuantAgent Original System", quant_summary)

    if kuant_summary is not None:
        pure_acc = pure_summary["accuracy_report"]["accuracy"]
        kuant_acc = kuant_summary["accuracy_report"]["accuracy"]
        lines.append("## KuantAgent vs Pure Algorithm")
        lines.append(f"- Accuracy delta: `{round(kuant_acc - pure_acc, 4)}`")
        if kuant_summary["accuracy_report"]["system_failure_count"] > 0:
            lines.append("- Fairness note: KuantAgent had system failures in this run, so direct comparison is not valid yet.")
        lines.append("")

    if kuant_summary is not None and quant_summary is not None:
        kuant_acc = kuant_summary["accuracy_report"]["accuracy"]
        quant_acc = quant_summary["accuracy_report"]["accuracy"]
        lines.append("## KuantAgent vs QuantAgent")
        lines.append(f"- Accuracy delta: `{round(kuant_acc - quant_acc, 4)}`")
        if (
            kuant_summary["accuracy_report"]["system_failure_count"] > 0
            or quant_summary["accuracy_report"]["system_failure_count"] > 0
        ):
            lines.append("- Fairness note: at least one full system had execution failures, so this comparison should not be used as a paper result.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrated benchmark comparison for pure algorithm, KuantAgent, and QuantAgent.")
    parser.add_argument("--benchmark-dir", required=True, help="Directory like benchmark/1h/btc")
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.15)
    parser.add_argument("--run-kuant-full", action="store_true")
    parser.add_argument("--run-quant-full", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N CSV files for quick validation")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based inclusive start position in the sorted CSV list.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="1-based inclusive end position in the sorted CSV list. 0 means no explicit end.",
    )
    parser.add_argument(
        "--system-timeout-sec",
        type=int,
        default=1800,
        help="Timeout in seconds for each full-system sample subprocess.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    csv_files = sorted(benchmark_dir.rglob("*.csv"), key=benchmark_sort_key)
    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")
    start_offset = args.start_index - 1
    if start_offset >= len(csv_files):
        raise ValueError(
            f"--start-index {args.start_index} is beyond the available sample count ({len(csv_files)})."
        )
    csv_files = csv_files[start_offset:]
    if args.end_index > 0:
        if args.end_index < args.start_index:
            raise ValueError("--end-index must be >= --start-index")
        explicit_count = args.end_index - args.start_index + 1
        csv_files = csv_files[:explicit_count]
    if args.limit > 0:
        csv_files = csv_files[: args.limit]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {benchmark_dir}")

    root_dir = Path(__file__).resolve().parents[1]
    kuant_root = root_dir / "KuantAgent"
    quant_root = root_dir / "QuantAgent"
    helper_script = Path(__file__).resolve().parent / "run_full_graph_sample.py"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "results" / f"integrated_comparison_{args.timeframe}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runner] Output directory: {output_dir}", file=sys.stderr, flush=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pure_results: List[Dict[str, Any]] = []
    kuant_results: List[Dict[str, Any]] = []
    quant_results: List[Dict[str, Any]] = []
    system_failures: List[Dict[str, Any]] = []

    total = len(csv_files)
    try:
        for index, csv_path in enumerate(csv_files, start=1):
            asset = infer_asset_from_filename(csv_path)
            print(f"[{index}/{total}] Running pure baseline for {csv_path.name}", flush=True)

            pure_raw = run_algorithmic_baseline(
                csv_path=csv_path,
                asset=asset,
                timeframe=args.timeframe,
                window_size=args.window_size,
                future_horizon=args.future_horizon,
                neutral_threshold_pct=args.neutral_threshold_pct,
            )
            pure_record = _to_eval_record(
                {
                    **pure_raw["baseline_result"],
                    "parsed_decision": pure_raw["analysis_output"].get("decision_features", {}),
                    "raw_decision": "",
                }
            )
            pure_results.append(pure_record)
            _checkpoint_output(output_dir, pure_results, kuant_results, quant_results)

            if args.run_kuant_full:
                print(f"[{index}/{total}] Running KuantAgent full system for {csv_path.name}", flush=True)
                try:
                    kuant_raw = _run_full_system_subprocess(
                        helper_script=helper_script,
                        project_root=kuant_root,
                        csv_path=csv_path,
                        asset=asset,
                        timeframe=args.timeframe,
                        window_size=args.window_size,
                        future_horizon=args.future_horizon,
                        neutral_threshold_pct=args.neutral_threshold_pct,
                        timeout_sec=args.system_timeout_sec,
                    )
                    kuant_results.append(_to_eval_record(kuant_raw))
                except Exception as exc:
                    error_text = str(exc)
                    print(
                        f"[runner] KuantAgent failed on {csv_path.name}; recording non-actionable placeholder and continuing.",
                        file=sys.stderr,
                        flush=True,
                    )
                    kuant_results.append(
                        _failure_eval_record(
                            asset=asset,
                            timeframe=args.timeframe,
                            sample_file=str(csv_path),
                            true_direction=pure_record["true_direction"],
                            future_return_pct=float(pure_record["future_return_pct"]),
                            is_neutral_move=int(pure_record["is_neutral_move"]),
                            horizon_total_count=int(pure_record["horizon_total_count"]),
                            future_step_directions=list(pure_record.get("future_step_directions", [])),
                            error=error_text,
                        )
                    )
                    system_failures.append(
                        {
                            "system": "kuant_full",
                            "sample_file": str(csv_path),
                            "error": error_text,
                        }
                    )
                _checkpoint_output(output_dir, pure_results, kuant_results, quant_results)

            if args.run_quant_full:
                print(f"[{index}/{total}] Running QuantAgent full system for {csv_path.name}", flush=True)
                try:
                    quant_raw = _run_full_system_subprocess(
                        helper_script=helper_script,
                        project_root=quant_root,
                        csv_path=csv_path,
                        asset=asset,
                        timeframe=args.timeframe,
                        window_size=args.window_size,
                        future_horizon=args.future_horizon,
                        neutral_threshold_pct=args.neutral_threshold_pct,
                        timeout_sec=args.system_timeout_sec,
                    )
                    quant_results.append(_to_eval_record(quant_raw))
                except Exception as exc:
                    error_text = str(exc)
                    print(
                        f"[runner] QuantAgent failed on {csv_path.name}; recording non-actionable placeholder and continuing.",
                        file=sys.stderr,
                        flush=True,
                    )
                    quant_results.append(
                        _failure_eval_record(
                            asset=asset,
                            timeframe=args.timeframe,
                            sample_file=str(csv_path),
                            true_direction=pure_record["true_direction"],
                            future_return_pct=float(pure_record["future_return_pct"]),
                            is_neutral_move=int(pure_record["is_neutral_move"]),
                            horizon_total_count=int(pure_record["horizon_total_count"]),
                            future_step_directions=list(pure_record.get("future_step_directions", [])),
                            error=error_text,
                        )
                    )
                    system_failures.append(
                        {
                            "system": "quant_full",
                            "sample_file": str(csv_path),
                            "error": error_text,
                        }
                    )
                _checkpoint_output(output_dir, pure_results, kuant_results, quant_results)
    except Exception as exc:
        _checkpoint_output(output_dir, pure_results, kuant_results, quant_results)
        if system_failures:
            (output_dir / "system_failures.json").write_text(
                json.dumps(system_failures, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_pure_samples": len(pure_results),
                    "completed_kuant_samples": len(kuant_results),
                    "completed_quant_samples": len(quant_results),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise

    if system_failures:
        (output_dir / "system_failures.json").write_text(
            json.dumps(system_failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report_text = _build_report(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        pure_summary=_summarize_system(pure_results),
        kuant_summary=_summarize_system(kuant_results) if kuant_results else None,
        quant_summary=_summarize_system(quant_results) if quant_results else None,
    )
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")

    combined_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    (output_dir / "summary.json").write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {"output_dir": str(output_dir), "summary": combined_summary},
            ensure_ascii=True,
            indent=2,
        )
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
