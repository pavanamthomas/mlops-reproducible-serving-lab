"""Reproducible training, local serving, and training-serving skew laboratory.

The public functions are small enough to be checked: a known DGP, a
fitted sklearn Pipeline, a FastAPI contract, and deliberate failure
cases. Nothing in this package is an empirical finding about a real
applicant population, and the Docker image is not a production service.
"""

from mlserv.artifacts import ArtifactBundle, load_bundle, save_bundle
from mlserv.config import TrainConfig, load_config
from mlserv.data import generate_dataset
from mlserv.schema import FEATURE_ORDER, SCHEMA_VERSION
from mlserv.train import train, train_and_persist

__version__ = "0.1.0"

__all__ = [
    "FEATURE_ORDER",
    "SCHEMA_VERSION",
    "ArtifactBundle",
    "TrainConfig",
    "generate_dataset",
    "load_bundle",
    "load_config",
    "save_bundle",
    "train",
    "train_and_persist",
]
