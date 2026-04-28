# KuantAgent Codebase

This repository contains the runnable code for a comparative trading-agent research project built around three evaluation paths:

- `Pure Algorithm`: a deterministic baseline executed through the benchmark pipeline
- `KuantAgent`: an algorithm-guided, AI-enhanced trading agent under active iteration
- `QuantAgent`: the reference-paper model kept as the comparison target

The repository root is the full `Code` workspace rather than a single model folder.

## Repository Structure

- `KuantAgent/`
  The main research model. It contains feature extraction, graph orchestration, indicator/pattern/trend/decision agents, and the current arbitration-oriented decision logic.

- `QuantAgent/`
  The reference model used for benchmark comparison. Its core strategy logic is preserved as much as possible for fair experiments.

- `benchmark/`
  Unified benchmark scripts, GUI runner, plotting tools, and the shared sample datasets used to evaluate all models on the same raw data.

- `参考文献/`, `学习日志/`, `毕设正文/`
  Local research materials and writing assets. These are intentionally excluded from the current public code push.

## Benchmark Workflow

The benchmark is designed so that all models receive the same raw sample files, while each model handles its own internal preprocessing and decision flow.

Main entry points:

- `benchmark/benchmark_gui.py`
  GUI launcher for integrated experiments

- `benchmark/run_integrated_comparison.py`
  Unified benchmark orchestrator

- `benchmark/run_full_graph_sample.py`
  Runs one full-agent sample for `KuantAgent` or `QuantAgent`

- `benchmark/run_pure_algo_experiment.py`
  Runs the pure algorithm baseline

## Current Design Direction

The current KuantAgent implementation focuses on:

- structured geometric representation of price action
- explicit structure-confirmation reasoning
- controlled AI intervention over an algorithmic base case
- arbitration between algorithmic and structural evidence in hard samples

The project goal is not only to compare three models fairly, but also to improve KuantAgent so it can absorb the stability of pure algorithms and the structure sensitivity of QuantAgent.

## Notes

- Benchmark result folders under `benchmark/results/` are excluded from version control.
- Research notes, thesis drafts, and auxiliary text materials are also excluded from this push.
- This repository currently prioritizes executable code, benchmark scripts, and shared experiment datasets.
