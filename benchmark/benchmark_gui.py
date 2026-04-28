from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "benchmark"
KUANT_DIR = ROOT_DIR / "KuantAgent"
DEFAULT_BENCHMARK_DIR = BENCHMARK_DIR / "1h" / "btc"
INTEGRATED_SCRIPT = BENCHMARK_DIR / "run_integrated_comparison.py"
PLOT_SCRIPT = BENCHMARK_DIR / "plot_integrated_comparison.py"


class BenchmarkRunnerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KuantAgent Benchmark Runner")
        self.root.geometry("1180x820")

        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False
        self.stdout_lines: list[str] = []
        self.last_output_dir = ""
        self.log_history: list[str] = []
        self.session_log_path = BENCHMARK_DIR / "results" / "gui_runner_latest.log"
        self.run_log_path: Path | None = None

        self.python_var = tk.StringVar(value=sys.executable)
        self.benchmark_dir_var = tk.StringVar(value=str(DEFAULT_BENCHMARK_DIR))
        self.timeframe_var = tk.StringVar(value="1h")
        self.window_size_var = tk.StringVar(value="45")
        self.future_horizon_var = tk.StringVar(value="3")
        self.neutral_threshold_var = tk.StringVar(value="0.15")
        self.start_index_var = tk.StringVar(value="1")
        self.end_index_var = tk.StringVar(value="")
        self.limit_var = tk.StringVar(value="10")
        self.timeout_var = tk.StringVar(value="1800")
        self.run_kuant_var = tk.BooleanVar(value=True)
        self.run_quant_var = tk.BooleanVar(value=True)
        self.auto_plot_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Ready")
        self.phase_var = tk.StringVar(value="Idle")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.command_preview_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="")
        self.figures_dir_var = tk.StringVar(value="")
        self.sample_progress_var = tk.StringVar(value="No sample running")
        self.system_progress_var = tk.StringVar(value="No system running")

        self._build_ui()
        self._refresh_command_preview()
        self.root.after(120, self._poll_log_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(container)
        top.pack(fill=tk.X)

        left = ttk.LabelFrame(top, text="Run Configuration", padding=10)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        right = ttk.LabelFrame(top, text="Live Status", padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(12, 0))

        self._add_labeled_entry(left, 0, "Python Executable", self.python_var, browse="file")
        self._add_labeled_entry(left, 1, "Benchmark Directory", self.benchmark_dir_var, browse="dir")
        self._add_labeled_entry(left, 2, "Timeframe", self.timeframe_var)
        self._add_labeled_entry(left, 3, "Window Size", self.window_size_var)
        self._add_labeled_entry(left, 4, "Future Horizon", self.future_horizon_var)
        self._add_labeled_entry(left, 5, "Neutral Threshold %", self.neutral_threshold_var)
        self._add_labeled_entry(left, 6, "Start Sample Index", self.start_index_var)
        self._add_labeled_entry(left, 7, "End Sample Index (optional)", self.end_index_var)
        self._add_labeled_entry(left, 8, "Sample Count", self.limit_var)
        self._add_labeled_entry(left, 9, "Per-System Timeout (sec)", self.timeout_var)

        checks = ttk.Frame(left)
        checks.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(checks, text="Run KuantAgent", variable=self.run_kuant_var, command=self._refresh_command_preview).pack(side=tk.LEFT)
        ttk.Checkbutton(checks, text="Run QuantAgent", variable=self.run_quant_var, command=self._refresh_command_preview).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(checks, text="Auto Plot After Run", variable=self.auto_plot_var).pack(side=tk.LEFT, padx=(12, 0))

        button_row = ttk.Frame(left)
        button_row.grid(row=11, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.start_button = ttk.Button(button_row, text="Start Run", command=self.start_run)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(button_row, text="Stop Run", command=self.stop_run, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Copy Command", command=self.copy_command).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(right, text="Overall Status").pack(anchor="w")
        ttk.Label(right, textvariable=self.status_var, wraplength=320).pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="Current Phase").pack(anchor="w")
        ttk.Label(right, textvariable=self.phase_var, wraplength=320).pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="Current Sample").pack(anchor="w")
        ttk.Label(right, textvariable=self.sample_progress_var, wraplength=320).pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="Current System").pack(anchor="w")
        ttk.Label(right, textvariable=self.system_progress_var, wraplength=320).pack(anchor="w", pady=(0, 8))
        ttk.Progressbar(right, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(right, text="Last Output Directory").pack(anchor="w")
        ttk.Label(right, textvariable=self.output_dir_var, wraplength=320).pack(anchor="w")
        ttk.Label(right, text="Last Figures Directory").pack(anchor="w", pady=(8, 0))
        ttk.Label(right, textvariable=self.figures_dir_var, wraplength=320).pack(anchor="w")

        preview_frame = ttk.LabelFrame(container, text="Command Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(12, 0))
        preview = tk.Text(preview_frame, height=4, wrap=tk.WORD)
        preview.pack(fill=tk.X, expand=False)
        preview.configure(state=tk.DISABLED)
        self.command_preview_widget = preview

        explain_frame = ttk.LabelFrame(container, text="What This Runner Does", padding=10)
        explain_frame.pack(fill=tk.X, pady=(12, 0))
        explain_text = (
            "1. run_integrated_comparison.py orchestrates the whole benchmark.\n"
            "2. baseline_runner.py evaluates the pure algorithm baseline on each sample.\n"
            "3. run_full_graph_sample.py launches one full KuantAgent or QuantAgent graph for one sample.\n"
            "4. You will see sample-level progress, helper-script launch commands, and inner graph stages such as "
            "Preparing sample, Initializing TradingGraph, and Invoking LangGraph workflow."
        )
        ttk.Label(explain_frame, text=explain_text, justify=tk.LEFT, wraplength=1120).pack(anchor="w")

        log_frame = ttk.LabelFrame(container, text="Live Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = tk.Text(log_frame, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_labeled_entry(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=78)
        entry.grid(row=row, column=1, sticky="ew", pady=4, padx=(8, 0))
        entry.bind("<KeyRelease>", lambda _event: self._refresh_command_preview())
        if browse == "file":
            ttk.Button(parent, text="Browse", command=lambda: self._browse_file(variable)).grid(row=row, column=2, padx=(8, 0))
        elif browse == "dir":
            ttk.Button(parent, text="Browse", command=lambda: self._browse_dir(variable)).grid(row=row, column=2, padx=(8, 0))
        parent.columnconfigure(1, weight=1)

    def _browse_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(initialdir=str(Path(variable.get()).parent if variable.get() else ROOT_DIR))
        if path:
            variable.set(path)
            self._refresh_command_preview()

    def _browse_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=variable.get() or str(ROOT_DIR))
        if path:
            variable.set(path)
            self._refresh_command_preview()

    def _build_command(self) -> list[str]:
        command = [
            self.python_var.get().strip(),
            str(INTEGRATED_SCRIPT),
            "--benchmark-dir",
            self.benchmark_dir_var.get().strip(),
            "--timeframe",
            self.timeframe_var.get().strip(),
            "--window-size",
            self.window_size_var.get().strip(),
            "--future-horizon",
            self.future_horizon_var.get().strip(),
            "--neutral-threshold-pct",
            self.neutral_threshold_var.get().strip(),
            "--start-index",
            self.start_index_var.get().strip() or "1",
            "--limit",
            self.limit_var.get().strip(),
            "--system-timeout-sec",
            self.timeout_var.get().strip(),
        ]
        end_index = self.end_index_var.get().strip()
        if end_index:
            command.extend(["--end-index", end_index])
        if self.run_kuant_var.get():
            command.append("--run-kuant-full")
        if self.run_quant_var.get():
            command.append("--run-quant-full")
        return command

    def _refresh_command_preview(self) -> None:
        command = subprocess.list2cmdline(self._build_command())
        self.command_preview_var.set(command)
        self.command_preview_widget.configure(state=tk.NORMAL)
        self.command_preview_widget.delete("1.0", tk.END)
        self.command_preview_widget.insert("1.0", command)
        self.command_preview_widget.configure(state=tk.DISABLED)

    def copy_command(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.command_preview_var.get())
        self.status_var.set("Command copied to clipboard")

    def start_run(self) -> None:
        if self.running:
            return
        if not Path(self.python_var.get().strip()).exists():
            messagebox.showerror("Invalid Python", "The selected Python executable does not exist.")
            return
        if not Path(self.benchmark_dir_var.get().strip()).exists():
            messagebox.showerror("Invalid Benchmark Directory", "The benchmark directory does not exist.")
            return
        try:
            start_index = int(self.start_index_var.get().strip() or "1")
            if start_index < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Start Index", "Start sample index must be an integer >= 1.")
            return
        end_index_text = self.end_index_var.get().strip()
        if end_index_text:
            try:
                end_index = int(end_index_text)
                if end_index < start_index:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid End Index", "End sample index must be blank or an integer >= start index.")
                return

        self.running = True
        self.stdout_lines = []
        self.log_history = []
        self.last_output_dir = ""
        self.run_log_path = None
        self.output_dir_var.set("")
        self.figures_dir_var.set("")
        self.sample_progress_var.set("Starting...")
        self.system_progress_var.set("Waiting for first stage...")
        self.progress_var.set(0.0)
        self.status_var.set("Starting benchmark...")
        self.phase_var.set("Launching process")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        try:
            self.session_log_path.write_text("", encoding="utf-8")
        except Exception:
            pass

        command = self._build_command()
        self._append_log("info", f"Launching benchmark orchestrator: {INTEGRATED_SCRIPT}")
        self._append_log("info", f"Command: {subprocess.list2cmdline(command)}")
        self._append_log(
            "info",
            "Pipeline: integrated comparison -> pure baseline + full KuantAgent graph + full QuantAgent graph",
        )

        self.reader_thread = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        self.reader_thread.start()

    def stop_run(self) -> None:
        if self.process and self.running:
            self._append_log("warn", "Stop requested by user. Terminating benchmark process...")
            self.process.terminate()
            self.status_var.set("Stopping benchmark...")

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if self.process.stdout is None:
                self.log_queue.put(("error", "Failed to capture subprocess stdout."))
                return

            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self.stdout_lines.append(line)
                self.log_queue.put(("line", line.rstrip("\n")))

            return_code = self.process.wait()
            if return_code == 0:
                self.log_queue.put(("done", f"Benchmark finished successfully with exit code {return_code}."))
                if self.auto_plot_var.get() and self.last_output_dir:
                    self._run_plotting(Path(self.last_output_dir))
            else:
                self.log_queue.put(("error", f"Benchmark exited with code {return_code}."))
        except Exception as exc:
            self.log_queue.put(("error", f"Failed to run benchmark process: {exc}"))
        finally:
            self.process = None
            self.log_queue.put(("finished", ""))

    def _run_plotting(self, results_dir: Path) -> None:
        plot_command = [
            self.python_var.get().strip(),
            str(PLOT_SCRIPT),
            "--results-dir",
            str(results_dir),
        ]
        self.log_queue.put(("phase", "Generating comparison figures"))
        self.log_queue.put(("info_line", f"Launching plotting script: {subprocess.list2cmdline(plot_command)}"))
        try:
            completed = subprocess.run(
                plot_command,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            self.log_queue.put(("error", f"Failed to launch plotting script: {exc}"))
            return

        stdout_text = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        if stdout_text:
            for line in stdout_text.splitlines():
                self.log_queue.put(("info_line", f"[plot] {line}"))
        if stderr_text:
            for line in stderr_text.splitlines():
                self.log_queue.put(("info_line", f"[plot][stderr] {line}"))

        if completed.returncode != 0:
            self.log_queue.put(("error", f"Plotting script failed with exit code {completed.returncode}."))
            return

        figures_dir = results_dir / "figures"
        self.log_queue.put(("figures_dir", str(figures_dir)))
        self.log_queue.put(("info_line", f"Figures generated in: {figures_dir}"))

    def _append_log(self, level: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_history.append(line)
        try:
            with self.session_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass
        if self.run_log_path is not None:
            try:
                with self.run_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:
                pass
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)

    def _persist_log_to_output_dir(self) -> None:
        if not self.last_output_dir:
            return
        try:
            output_log_path = Path(self.last_output_dir) / "gui_runner.log"
            output_log_path.write_text("\n".join(self.log_history) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _explain_line(self, line: str) -> str | None:
        if "Running pure baseline for" in line:
            return "调用 baseline_runner.py，执行纯算法基线，不调用大模型。"
        if "Running KuantAgent full system for" in line:
            return "调用 run_full_graph_sample.py，为当前样本启动 KuantAgent 全链路工作流。"
        if "Running QuantAgent full system for" in line:
            return "调用 run_full_graph_sample.py，为当前样本启动 QuantAgent 原版多模态工作流。"
        if "[runner] Launching" in line:
            return "外层 orchestrator 正在拉起单样本 helper 脚本。"
        if "Preparing sample" in line:
            return "helper 脚本正在切分 CSV，准备输入窗口和未来标签。"
        if "Structured-only mode enabled" in line:
            return "KuantAgent 当前走结构化主链，跳过图像生成。"
        if "Generating chart images" in line:
            return "当前系统需要图像输入，正在生成 K 线图和趋势图。"
        if "Initializing TradingGraph" in line:
            return "正在构建 LangGraph 工作流和模型节点。"
        if "Invoking LangGraph workflow" in line:
            return "开始执行完整 Agent 工作流，接下来是真正的大模型推理阶段。"
        if "Graph invocation completed" in line:
            return "单样本图工作流已完成，正在整理结果并返回给总控脚本。"
        return None

    def _handle_progress_line(self, line: str) -> None:
        match = re.match(r"^\[(\d+)/(\d+)\]\s+Running\s+(.+?)\s+for\s+(.+)$", line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            stage = match.group(3)
            sample = match.group(4)
            progress = (current - 1) / max(total, 1) * 100.0
            self.progress_var.set(progress)
            self.phase_var.set(f"{current}/{total} | {stage} | {sample}")
            self.status_var.set("Benchmark running")
            self.sample_progress_var.set(f"Sample {current}/{total}: {sample}")
            self.system_progress_var.set(stage)
            return

        if line.startswith("[runner] Launching"):
            self.phase_var.set("Launching sample helper")
        helper_match = re.match(r"^\[(KuantAgent|QuantAgent):(.+?)\]\s+(.*)$", line)
        if helper_match:
            system = helper_match.group(1)
            sample = helper_match.group(2)
            detail = helper_match.group(3)
            self.system_progress_var.set(f"{system} | {sample}")
            self.phase_var.set(detail[:120] if detail else f"{system} helper running")
        elif "Preparing sample" in line:
            self.phase_var.set("Preparing sample window and labels")
        elif "Generating chart images" in line:
            self.phase_var.set("Generating multimodal chart inputs")
        elif "Structured-only mode enabled" in line:
            self.phase_var.set("Structured-only mode: skipping charts")
        elif "Initializing TradingGraph" in line:
            self.phase_var.set("Initializing TradingGraph")
        elif "Invoking LangGraph workflow" in line:
            self.phase_var.set("Invoking LangGraph workflow")
        elif "Graph invocation completed" in line:
            self.phase_var.set("Single-sample workflow completed")

        output_dir_match = re.search(r'"output_dir"\s*:\s*"([^"]+)"', line)
        if output_dir_match:
            try:
                self.last_output_dir = json.loads(f"\"{output_dir_match.group(1)}\"")
            except Exception:
                self.last_output_dir = output_dir_match.group(1)
            self.output_dir_var.set(self.last_output_dir)
            self.run_log_path = Path(self.last_output_dir) / "gui_runner.log"
            self._persist_log_to_output_dir()
            return

        runner_output_match = re.search(r"^\[runner\]\s+Output directory:\s+(.+)$", line)
        if runner_output_match:
            self.last_output_dir = runner_output_match.group(1).strip()
            self.output_dir_var.set(self.last_output_dir)
            self.run_log_path = Path(self.last_output_dir) / "gui_runner.log"
            self._persist_log_to_output_dir()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self._append_log("info", payload)
                    self._handle_progress_line(payload)
                    explanation = self._explain_line(payload)
                    if explanation:
                        self._append_log("hint", f"  -> {explanation}")
                elif kind == "done":
                    self.status_var.set(payload)
                    self.progress_var.set(100.0)
                    self._append_log("ok", payload)
                    self._persist_log_to_output_dir()
                elif kind == "phase":
                    self.phase_var.set(payload)
                    self._append_log("info", payload)
                elif kind == "info_line":
                    self._append_log("info", payload)
                elif kind == "error":
                    self.status_var.set("Benchmark failed")
                    self._append_log("error", payload)
                    self._persist_log_to_output_dir()
                elif kind == "figures_dir":
                    self.figures_dir_var.set(payload)
                    self._persist_log_to_output_dir()
                elif kind == "finished":
                    self.running = False
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    if self.last_output_dir:
                        self.output_dir_var.set(self.last_output_dir)
                        self._append_log("info", f"Latest result directory: {self.last_output_dir}")
                        self._persist_log_to_output_dir()
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_log_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass
    BenchmarkRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
