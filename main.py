import os
import argparse

from config import FACESCAPE_CONFIG as cfg

import torch
from src.datasets.dataset import Faces3D
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from src.models.model import Model_2S_SGCN
from src.utils.log import initialise_log
from src.utils.utils import NormalizeScale
from src.scripts.train import train
from src.scripts.test import test
from src.scripts.refine import refine


def main(do_refine=True):
    device = torch.device(cfg.device)
    dataset = Faces3D(root=".data", pre_transform=NormalizeScale(), n_resampling_points=cfg.n_resampling_points)

    model = Model_2S_SGCN(cfg.model["in_channels"], cfg.model["hidden_channels"], cfg.model["n_landmarks"], cfg.model["depth"]).to(device)
    parameters = list(model.parameters())
    optimizer = cfg.optimizer["class"](parameters, **cfg.optimizer["params"])
    criterion = cfg.loss["class"](**cfg.loss["params"])

    train_dataset, test_dataset = random_split(dataset, [cfg.split["train"], cfg.split["test"]], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=cfg.data_loader["training"]["batch_size"], shuffle=cfg.data_loader["training"]["shuffle"])
    test_loader = DataLoader(test_dataset, batch_size=cfg.data_loader["testing"]["batch_size"], shuffle=cfg.data_loader["testing"]["shuffle"])

    writer, models_path = initialise_log(cfg)

    for epoch in range(1, cfg.epochs + 1):
        model = train(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            top_t=cfg.model["top_t"],
            sigma=cfg.model["distance_sigma"],
            device=device,
            K_KNN=cfg.model["k_knn"],
            NEUTRAL_MEAN_SCALE=dataset[0].mean_scale,
            log=True,
            writer=writer,
        )

        test(
            epoch,
            model,
            test_loader,
            criterion,
            top_t=cfg.model["top_t"],
            sigma=cfg.model["distance_sigma"],
            device=device,
            K_KNN=cfg.model["k_knn"],
            NEUTRAL_MEAN_SCALE=dataset[0].mean_scale,
            log=True,
            writer=writer,
        )

        if epoch % cfg.saving_step == 0:
            torch.save(model, os.path.join(models_path, f"{cfg.model['name']}_{epoch}.pt"))
        torch.cuda.empty_cache()

    if do_refine:
        models_dir = os.path.dirname(models_path)
        refine_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        refine(
            model,
            refine_loader=refine_loader,
            results_folder=models_dir,
            top_t=cfg.model["top_t"],
            sigma=cfg.model["distance_sigma"],
            gamma=cfg.model["gamma"],
            device=device,
            K_KNN=cfg.model["k_knn"],
            log=True,
        )


if __name__ == "__main__":
    # do_refine from sys.args
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_refine", type=bool, default=True)
    args = parser.parse_args()
    main(args.do_refine)
