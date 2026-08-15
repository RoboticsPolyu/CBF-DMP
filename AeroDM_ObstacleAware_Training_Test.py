"""Compatibility launcher for the modular :mod:`aerodm_core` package.

Existing imports from this historical script continue to work, while all
implementation now lives in focused package modules.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Public compatibility exports. Keeping Config in this module also allows old
# checkpoints that pickle ``__main__.Config`` to load when this script is run.
from aerodm_core.config import Config
from aerodm_core.data import *  # noqa: F401,F403
from aerodm_core.diffusion import ObstacleAwareDiffusionProcess
from aerodm_core.evaluation import test_model_performance_cb_eva
from aerodm_core.experiment import run_experiment
from aerodm_core.guidance import (compute_barrier_and_grad,
                                  compute_barrier_and_grad_logistic)
from aerodm_core.losses import AeroDMLoss
from aerodm_core.metrics import *  # noqa: F401,F403
from aerodm_core.model import AeroDM
from aerodm_core.models import (AttentionObstacleEncoder, ConditionEmbedding,
                                ObstacleAwareDiffusionTransformer,
                                ObstacleEncoder, PositionalEncoding)
from aerodm_core.qualitative import generate_trj_demos, test_action_effect
from aerodm_core.reporting import *  # noqa: F401,F403
from aerodm_core.visualization import *  # noqa: F401,F403


if __name__ == "__main__":
    run_experiment(Config())
