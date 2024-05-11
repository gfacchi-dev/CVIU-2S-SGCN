import os
import argparse

from config import FACESCAPE_CONFIG as cfg

import torch
from src.datasets.dataset import Faces3D
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from src.models.model import Model_2S_SGCN
from src.utils.utils import NormalizeScale
from src.scripts.refine import refine


def main(model_path):
    device = torch.device(cfg.device)
    dataset = Faces3D(root=".data", pre_transform=NormalizeScale(), n_resampling_points=cfg.n_resampling_points)

    model = torch.load(model_path)
    _, test_dataset = random_split(dataset, [cfg.split["train"], cfg.split["test"]], generator=torch.Generator().manual_seed(42))

    refine_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    refine(
        model,
        refine_loader=refine_loader,
        results_folder=os.path.dirname(os.path.dirname(model_path)),
        top_t=cfg.model["top_t"],
        device=device,
        K_KNN=cfg.model["k_knn"],
        n_resampling_points=cfg.n_resampling_points,
        log=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model to refine")
    args = parser.parse_args()
    path = args.model_path

    main(path)
