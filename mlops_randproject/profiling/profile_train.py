# profile_train.py

import cProfile
import sys
from pathlib import Path
import os


def main():
    # profile_train.py (update)
    project_root = Path(__file__).resolve().parents[2]  # go up to se489-mlops
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)
    print(f"[DEBUG] PYTHONPATH set to: {project_root}")

    # Command-line overrides (e.g., model=mlp train.epochs=2)
    # overrides = " ".join(sys.argv[1:])
    module_command = "import runpy; runpy.run_module('mlops_randproject.models.model_training', run_name='__main__')"

    output_file = Path(__file__).parent / "train_profile.prof"
    print(f" Profiling training script... Output: {output_file}")

    cProfile.run(module_command, filename=str(output_file))

    print(" Profiling complete. View with:")
    print(f"  python -m pstats {output_file}")


if __name__ == "__main__":
    main()
