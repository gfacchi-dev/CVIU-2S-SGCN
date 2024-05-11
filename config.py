from dataclasses import dataclass
import torch

from src.models.loss import AdaptiveWingLoss
from torch.optim import Adam


@dataclass
class FACESCAPE_CONFIG:
    device: str = "cuda"
    n_resampling_points: int = 32768
    model = {
        "name": "Model_2S_SGCN",
        "depth": 4,
        "in_channels": 6,
        "hidden_channels": 64,
        "n_landmarks": 68,
        "k_knn": 32,
        "top_t": 7,
        "distance_sigma": 0.03,
    }
    optimizer = {
        "class": Adam,
        "params": {
            "lr": 0.001,
        },
    }
    loss = {
        "class": AdaptiveWingLoss,
        "params": {
            "theta": 0.5,
        },
    }
    split = {
        "train": 0.8,
        "test": 0.2,
    }
    data_loader = {
        "training": {
            "batch_size": 2,
            "shuffle": True,
        },
        "testing": {
            "batch_size": 2,
            "shuffle": False,
        },
    }

    epochs: int = 1
    saving_step: int = 1
