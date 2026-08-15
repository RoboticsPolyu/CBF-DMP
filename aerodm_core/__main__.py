"""Run AeroDM with ``python -m aerodm_core``."""

# Keep Config in __main__ for checkpoints produced by the legacy script.
from .config import Config
from .experiment import run_experiment


if __name__ == "__main__":
    run_experiment(Config())
