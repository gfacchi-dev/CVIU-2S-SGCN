from tqdm import tqdm
import torch
import numpy as np
from torch_cluster import knn_graph
from src.utils.utils import gaussian_distance


def log_test(writer, epoch, loss, mean, std, mean_of_top_t_distance, std_of_top_t_distance, mean_of_ponderate_distance, std_of_ponderate_distance):
    writer.add_scalar("Loss/test", loss, epoch)
    writer.add_scalar("GD/test", mean, epoch)
    writer.add_scalar("Std GD/test", std, epoch)
    writer.add_scalar("TKD/test", mean_of_top_t_distance, epoch)
    writer.add_scalar("Std TKD/test", std_of_top_t_distance, epoch)
    writer.add_scalar("PD/test", mean_of_ponderate_distance, epoch)
    writer.add_scalar("Std PD/test", std_of_ponderate_distance, epoch)
    writer.flush()


@torch.no_grad()
def test(epoch, model, test_loader, criterion, top_t=10, sigma=0.03, device="cuda", K_KNN=32, NEUTRAL_MEAN_SCALE=1.0, log=False, writer=None):
    model.eval()
    mean_gaus_distances = []
    std_gaus_distances = []
    mean_top_k_distances = []
    std_top_k_distances = []
    losses = []
    mean_ponderata_distances = []
    std_ponderata_distances = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, data in pbar:
        data = data.to(device)
        batch_size = int(data.num_graphs)
        pos = data.pos
        normal = data.normal
        batch = data.batch
        landmarks = data.landmarks
        super_indexes = data.super_indexes
        heatmap_gts = gaussian_distance(data.distances, sigma).to(device)

        num_super_indexes = super_indexes.shape[0] // batch_size
        local_edges = knn_graph(pos, k=K_KNN, batch=batch, loop=True)
        global_edges = torch.zeros((2, (num_super_indexes**2) * batch_size), dtype=torch.long, device=device)
        super_indexes = torch.reshape(super_indexes, (-1, num_super_indexes))
        for j in range(0, batch_size):
            global_edges[:, j * num_super_indexes**2 : (j + 1) * num_super_indexes**2] = torch.cartesian_prod(super_indexes[j, :], super_indexes[j, :]).T

        x = torch.cat([pos, normal], dim=-1)
        out, _ = model(x, global_edges, local_edges)  # 8*32k X 68
        loss_m1 = criterion(out, heatmap_gts)

        interpr = out.reshape((batch_size, int(out.shape[0] / batch_size), out.shape[1]))  # 8 x 32k x 68

        # emb_for_refine = emb_for_refine.reshape((batch_size, int(emb_for_refine.shape[0]/batch_size), emb_for_refine.shape[1])) # 8 x 32k x 64
        # 2 x 32k x 64
        max_values, max_indices = torch.topk(interpr, top_t, dim=1)  # 2 x 10 x 68

        # max_indices_emb = max_indices.repeat_interleave(64, dim=2).reshape(batch_size, -1, 64) # 2 x 680 x 64
        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)  # 2 x 680 x 3

        max_values, max_indices = torch.topk(interpr, top_t, dim=1)
        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)
        max_index = torch.argmax(interpr, dim=1)
        max_index = max_index.repeat_interleave(3, dim=1).view(batch_size, -1, 3)

        pos = pos.reshape(batch_size, -1, 3)
        max_positions = pos.gather(1, max_indices).reshape(batch_size, top_t, -1, 3)
        mean_of_max_positions = torch.mean(max_positions, dim=1)
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

    mean_gaus_distance = np.mean(mean_gaus_distances)
    std_gaus_distance = np.mean(std_gaus_distances)
    mean_losses = np.mean(losses)
    mean_of_top_k_distance = np.mean(mean_top_k_distances)
    std_of_top_k_distance = np.mean(std_top_k_distances)
    mean_of_ponderate_distance = np.mean(mean_ponderata_distances)
    std_of_ponderate_distance = np.mean(std_ponderata_distances)

    if log:
        log_test(writer, epoch, mean_losses, mean_gaus_distance, std_gaus_distance, mean_of_top_k_distance, std_of_top_k_distance, mean_of_ponderate_distance, std_of_ponderate_distance)

    return mean_losses, mean_gaus_distance, std_gaus_distance, mean_of_top_k_distance, std_of_top_k_distance, mean_of_ponderate_distance, std_of_ponderate_distance
