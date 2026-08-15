# AeroDM Core Architecture

The package follows a one-way dependency flow:

```text
config
  ├── models ── model
  ├── guidance ── diffusion ── model
  ├── data
  ├── metrics ── plotting
  └── losses

data + model + visualization + reporting
                ↓
            evaluation
                ↓
            experiment
```

## Modules

- `config.py`: experiment defaults only.
- `models.py`: transformer and conditioning network components.
- `guidance.py`: barrier values and analytic gradients.
- `diffusion.py`: forward/reverse DDPM process.
- `model.py`: high-level `AeroDM` sampling interface.
- `losses.py`: training losses.
- `data.py`: normalization, target, and obstacle utilities.
- `metrics.py`: collision, error, and success metrics.
- `visualization/trajectory.py`: low-level trajectory and obstacle plots.
- `visualization/results.py`: composite evaluation figures.
- `visualization/styles.py`: maneuver-style and statistics figures.
- `reporting.py`: console/file reports and progress formatting.
- `evaluation.py`: quantitative evaluation workflow.
- `qualitative.py`: trajectory demos and maneuver-style experiments.
- `experiment.py`: reusable train/load/evaluate orchestration.

Run the project with either:

```bash
python AeroDM_ObstacleAware_Training_Test.py
python -m aerodm_core
```

For programmatic use:

```python
from aerodm_core import Config, AeroDM
from aerodm_core.experiment import prepare_datasets, run_experiment
```
