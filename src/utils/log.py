import os
import time
from torch.utils.tensorboard import SummaryWriter


def initialise_log(cfg):
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = os.path.join(".experiments", cfg.model["name"], date_time)

    if not os.path.exists(os.path.join(".experiments", cfg.model["name"])):
        os.makedirs(os.path.join(".experiments", cfg.model["name"]))

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(folder_name)
    models_path = os.path.join(folder_name, "models")

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    return writer, models_path
