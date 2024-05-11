from tqdm import tqdm
import torch
import numpy as np
from torch_cluster import knn_graph
from src.utils.utils import gaussian_distance


def log_train(writer, epoch, loss, mean, std, mean_of_top_t_distance, std_of_top_t_distance, mean_of_ponderate_distance, std_of_ponderate_distance):
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("GD/train", mean, epoch)
    writer.add_scalar("Std GD/train", std, epoch)
    writer.add_scalar("TKD/train", mean_of_top_t_distance, epoch)
    writer.add_scalar("Std TKD/train", std_of_top_t_distance, epoch)
    writer.add_scalar("PD/train", mean_of_ponderate_distance, epoch)
    writer.add_scalar("Std PD/train", std_of_ponderate_distance, epoch)
    writer.flush()


def train(epoch, model, train_loader, optimizer, criterion, top_t=10, distance_sigma=0.03, device="cuda", K_KNN=32, NEUTRAL_MEAN_SCALE=1.0, log=False, writer=None):
    model.train()
    mean_gaus_distances = []
    std_gaus_distances = []
    mean_top_k_distances = []
    std_top_k_distances = []
    mean_ponderata_distances = []
    std_ponderata_distances = []
    losses = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        data = data.to(device)
        batch_size = int(data.num_graphs)
        pos = data.pos
        normal = data.normal
        batch = data.batch
        landmarks = data.landmarks
        super_indexes = data.super_indexes
        heatmap_gts = gaussian_distance(data.distances, distance_sigma).to(device)

        num_super_indexes = super_indexes.shape[0] // batch_size
        local_edges = knn_graph(pos, k=K_KNN, batch=batch, loop=True)
        global_edges = torch.zeros((2, (num_super_indexes**2) * batch_size), dtype=torch.long, device=device)
        super_indexes = torch.reshape(super_indexes, (-1, num_super_indexes))
        for j in range(0, batch_size):
            global_edges[:, j * num_super_indexes**2 : (j + 1) * num_super_indexes**2] = torch.cartesian_prod(super_indexes[j, :], super_indexes[j, :]).T

        x = torch.cat([pos, normal], dim=-1)
        optimizer.zero_grad()
        out, _ = model(x, global_edges, local_edges)  # 8*32k X 68
        loss_m1 = criterion(out, heatmap_gts)

        interpr = out.reshape((batch_size, int(out.shape[0] / batch_size), out.shape[1]))  # 8 x 32k x 68
        max_values, max_indices = torch.topk(interpr, top_t, dim=1)  # 2 x 10 x 68

        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)  # 2 x 680 x 3

        max_index = torch.argmax(interpr, dim=1)
        max_index = max_index.repeat_interleave(3, dim=1).view(batch_size, -1, 3)

        pos = pos.reshape(batch_size, -1, 3)
        max_positions = pos.gather(1, max_indices).reshape(batch_size, top_t, -1, 3)  # Nella -1 ci va il numero di landmarks
        mean_of_max_positions = torch.mean(max_positions, dim=1)  # BATCH_SIZE x LMK x 3
        gaussian_max = pos.gather(1, max_index)
        loss = loss_m1

        coeffs = torch.softmax(max_values, dim=1)
        numeratore = torch.einsum("ijkm,ijk->ijkm", max_positions, coeffs)
        pred_lmk_pos_ponderata = torch.sum(numeratore, dim=1)

        gaussian_max = gaussian_max.reshape(-1, 3)
        mean_of_max_positions = mean_of_max_positions.reshape(-1, 3)
        pred_lmk_pos_ponderata = pred_lmk_pos_ponderata.reshape(-1, 3)
        mean_gaus_distance = torch.mean(torch.linalg.norm(gaussian_max - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE
        std_gaus_distance = torch.std(torch.linalg.norm(gaussian_max - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE
        mean_of_top_k_distance = torch.mean(torch.linalg.norm(mean_of_max_positions - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE
        std_of_top_k_distance = torch.std(torch.linalg.norm(mean_of_max_positions - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE
        mean_of_ponderate = torch.mean(torch.linalg.norm(pred_lmk_pos_ponderata - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE
        std_of_ponderate = torch.std(torch.linalg.norm(pred_lmk_pos_ponderata - landmarks, dim=1)) / NEUTRAL_MEAN_SCALE

        mean_gaus_distances.append(mean_gaus_distance.item())
        std_gaus_distances.append(std_gaus_distance.item())
        mean_top_k_distances.append(mean_of_top_k_distance.item())
        std_top_k_distances.append(std_of_top_k_distance.item())
        mean_ponderata_distances.append(mean_of_ponderate.item())
        std_ponderata_distances.append(std_of_ponderate.item())

        losses.append(loss.item())

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "GD": f"{mean_gaus_distance.item():.2f}±{std_gaus_distance.item():.2f}",
                "TKD": f"{mean_of_top_k_distance.item():.2f}±{std_of_top_k_distance.item():.2f}",
                "PD": f"{mean_of_ponderate.item():.2f}±{std_of_ponderate.item():.2f}",
            }
        )
        loss.backward()
        optimizer.step()

    epoch_mean_loss = np.mean(losses)
    mean_gaus_distance = np.mean(mean_gaus_distances)
    std_gaus_distance = np.mean(std_gaus_distances)
    mean_of_top_k_distance = np.mean(mean_top_k_distances)
    std_of_top_k_distance = np.mean(std_top_k_distances)
    mean_of_ponderate_distance = np.mean(mean_ponderata_distances)
    std_of_ponderate_distance = np.mean(std_ponderata_distances)

    if log == True:
        log_train(writer, epoch, epoch_mean_loss, mean_gaus_distance, std_gaus_distance, mean_of_top_k_distance, std_of_top_k_distance, mean_of_ponderate_distance, std_of_ponderate_distance)

    return model
