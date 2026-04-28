from pathlib import Path
import runpy


if __name__ == "__main__":
    shared_script = Path(__file__).resolve().parents[1] / "KuantAgent" / "validate_qwen.py"
    runpy.run_path(str(shared_script), run_name="__main__")
