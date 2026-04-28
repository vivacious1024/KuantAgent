from pathlib import Path
import runpy


if __name__ == "__main__":
    shared_script = Path(__file__).resolve().parents[1] / "KuantAgent" / "test_key_model.py"
    runpy.run_path(str(shared_script), run_name="__main__")
