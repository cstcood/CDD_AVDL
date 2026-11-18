"""CDD-AVDL: Convenient Depression Diagnosis via Audio-Visual Deep Learning.

Reference implementation of

    L. Liu, S. Chen, K. Chen, J. Xu, X. Chen,
    "Catching the Blackdog Easily: A Convenient Depression Diagnosis Method
     based on Audio-Visual Deep Learning",
    IEEE Transactions on Affective Computing, 2025.
    DOI: 10.1109/TAFFC.2025.3571697
"""

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .factory import build_dataloaders, build_fold_loaders, build_model
from .models.cdd_avdl import CDDAVDL

__version__ = "1.0.0"
__all__ = [
    "CDDAVDL",
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "build_model",
    "build_dataloaders",
    "build_fold_loaders",
    "__version__",
]
