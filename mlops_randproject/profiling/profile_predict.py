import cProfile
import pstats
import sys
from pathlib import Path
import os

def main():
    # Set PYTHONPATH so mlops_randproject is importable
    repo_root = Path(__file__).resolve().parents[2]  # One level up from profiling/
    os.environ["PYTHONPATH"] = str(repo_root)
    print(f"[DEBUG] PYTHONPATH set to: {repo_root}")

    # Command-line overrides (e.g., model=mlp)
    overrides = " ".join(sys.argv[1:])
    module_command = f"import runpy; runpy.run_module('mlops_randproject.models.predict_model', run_name='__main__')"

    output_file = Path(__file__).parent / "predict_profile.prof"
    print(f" Profiling predict script... Output: {output_file}")

    cProfile.run(module_command, filename=str(output_file))

    print(" Profiling complete. View with:")
    print(f"  python -m pstats {output_file}")

if __name__ == "__main__":
    main()
