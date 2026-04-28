import importlib.util
import os
import sys
from pathlib import Path

import requests
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

EXPECTED_MODEL = os.environ.get("QWEN_TEST_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Thinking")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECTS = {
    "KuantAgent": ROOT_DIR / "KuantAgent",
    "QuantAgent": ROOT_DIR / "QuantAgent",
}


def mask_key(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def emit_result(results: list[dict], name: str, status: str, detail: str) -> None:
    results.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}")
    print(f"      {detail}")


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_environment(results: list[dict]) -> str:
    print_section("Step 1/5 - Check SiliconFlow environment")
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        emit_result(results, "Environment Variable", "FAIL", "SILICONFLOW_API_KEY is empty.")
        raise SystemExit("SILICONFLOW_API_KEY is empty.")

    detail = (
        f"SILICONFLOW_API_KEY loaded, length={len(api_key)}, "
        f"masked={mask_key(api_key)}, model={EXPECTED_MODEL}"
    )
    emit_result(results, "Environment Variable", "PASS", detail)
    return api_key


def check_project_configs(results: list[dict]) -> None:
    print_section("Step 2/5 - Check both project defaults")
    for project_name, project_root in PROJECTS.items():
        default_config_path = project_root / "default_config.py"
        module = load_module(f"{project_name}_default_config", default_config_path)
        config = getattr(module, "DEFAULT_CONFIG", {})
        mismatches = []
        for key in ("agent_llm_model", "graph_llm_model", "vision_llm_model"):
            if config.get(key) != EXPECTED_MODEL:
                mismatches.append(f"{key}={config.get(key)!r}")
        if mismatches:
            emit_result(
                results,
                f"{project_name} default_config",
                "FAIL",
                " ; ".join(mismatches),
            )
        else:
            emit_result(
                results,
                f"{project_name} default_config",
                "PASS",
                f"agent/graph/vision defaults are all set to {EXPECTED_MODEL}",
            )


def test_raw_http(api_key: str, results: list[dict]) -> None:
    print_section("Step 3/5 - Test SiliconFlow raw HTTP")
    url = f"{SILICONFLOW_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EXPECTED_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
        "max_tokens": 5,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            emit_result(results, "SiliconFlow Raw HTTP", "PASS", response.text[:200])
        else:
            emit_result(
                results,
                "SiliconFlow Raw HTTP",
                "FAIL",
                f"status={response.status_code}, body={response.text[:300]}",
            )
    except Exception as exc:
        emit_result(results, "SiliconFlow Raw HTTP", "FAIL", str(exc))


def test_langchain_openai(api_key: str, results: list[dict]) -> None:
    print_section("Step 4/5 - Test ChatOpenAI via SiliconFlow")
    try:
        llm = ChatOpenAI(
            model=EXPECTED_MODEL,
            api_key=api_key,
            base_url=SILICONFLOW_BASE_URL,
            temperature=0,
            max_retries=1,
        )
        response = llm.invoke("Reply with exactly one word: OK")
        content = getattr(response, "content", response)
        emit_result(results, "langchain_openai.ChatOpenAI", "PASS", str(content))
    except Exception as exc:
        emit_result(results, "langchain_openai.ChatOpenAI", "FAIL", str(exc))


def test_project_runtime_smoke(results: list[dict]) -> None:
    print_section("Step 5/5 - Test both project TradingGraph runtime wiring")
    for project_name, project_root in PROJECTS.items():
        trading_graph_path = project_root / "trading_graph.py"
        default_config_path = project_root / "default_config.py"
        try:
            default_config_module = load_module(
                f"{project_name}_default_config_runtime",
                default_config_path,
            )
            config = dict(getattr(default_config_module, "DEFAULT_CONFIG", {}))
            trading_graph_module = load_module(
                f"{project_name}_trading_graph",
                trading_graph_path,
            )
            TradingGraph = getattr(trading_graph_module, "TradingGraph")
            graph = TradingGraph(config=config)
            emit_result(
                results,
                f"{project_name} TradingGraph runtime",
                "PASS",
                (
                    f"provider={graph.config.get('agent_llm_provider')}, "
                    f"agent_model={graph.config.get('agent_llm_model')}"
                ),
            )
        except Exception as exc:
            emit_result(
                results,
                f"{project_name} TradingGraph runtime",
                "FAIL",
                str(exc),
            )


def print_summary(results: list[dict]) -> int:
    print_section("Summary")
    for item in results:
        print(f"{item['status']:>4} | {item['name']}")

    fail_count = sum(1 for item in results if item["status"] == "FAIL")
    skip_count = sum(1 for item in results if item["status"] == "SKIP")
    pass_count = sum(1 for item in results if item["status"] == "PASS")

    print("")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"SKIP: {skip_count}")

    if fail_count:
        print("\nConclusion: at least one key connectivity check failed. Inspect the FAIL lines above first.")
        return 1

    print("\nConclusion: key connectivity checks all passed. You can proceed to integrated experiments.")
    return 0


def main() -> None:
    results: list[dict] = []
    api_key = check_environment(results)
    check_project_configs(results)
    test_raw_http(api_key, results)
    test_langchain_openai(api_key, results)
    test_project_runtime_smoke(results)
    raise SystemExit(print_summary(results))


if __name__ == "__main__":
    main()
