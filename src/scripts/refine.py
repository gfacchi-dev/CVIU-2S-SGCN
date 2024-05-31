import os
import torch
from torch_cluster import knn_graph
from src.utils.utils import NormalizeScale, gaussian_distance
import potpourri3d as pp3d
import open3d as o3d

import numpy as np
import scipy
from tqdm.auto import tqdm
import time


# TODO: Optimize .pt creation in order to not apply the Rigid Transformation to each mesh again
@torch.no_grad()
def refine(model, refine_loader, results_folder, top_t=7, sigma=0.03, gamma=5, device="cuda", K_KNN=32, log=True):
    model.eval()
    pbar = tqdm(enumerate(refine_loader), total=len(refine_loader))
    ponds = []
    smarts = []
    for i, data in pbar:
        data = data.to(device)
        batch_size = int(data.num_graphs)
        pos = data.pos
        normal = data.normal
        batch = data.batch
        landmarks = data.landmarks
        super_indexes = data.super_indexes
        num_super_indexes = super_indexes.shape[0] // batch_size

        local_edges = knn_graph(pos, k=K_KNN, batch=batch, loop=True)
        global_edges = torch.zeros((2, (num_super_indexes**2) * batch_size), dtype=torch.long, device=device)
        super_indexes = torch.reshape(super_indexes, (-1, num_super_indexes))
        for j in range(0, batch_size):
            global_edges[:, j * num_super_indexes**2 : (j + 1) * num_super_indexes**2] = torch.cartesian_prod(super_indexes[j, :], super_indexes[j, :]).T

        x = torch.cat([pos, normal], dim=-1)
        out, _ = model(x, global_edges, local_edges)  # 8*32k X 68
        n_landmarks = out.shape[1]
        out = out.reshape((batch_size, int(out.shape[0] / batch_size), n_landmarks))  # 8 x 32k x 68

        max_values, max_indices = torch.topk(out, top_t, dim=1)  # 2 x 10 x 68

        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)  # 2 x 680 x 3

        max_values, max_indices = torch.topk(out, top_t, dim=1)
        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)

        max_index = torch.argmax(out, dim=1)
        max_index = max_index.repeat_interleave(3, dim=1).view(batch_size, -1, 3)

        pos = pos.reshape(batch_size, -1, 3)
        max_positions = pos.gather(1, max_indices).reshape(batch_size, top_t, -1, 3)  # Nella -1 ci va il numero di landmarks
        normals_reshaped = normal.reshape(batch_size, -1, 3)
        max_normals = normals_reshaped.gather(1, max_indices).reshape(batch_size, top_t, -1, 3)

        coeffs = torch.softmax(max_values, dim=1)
        numeratore = torch.einsum("ijkm,ijk->ijkm", max_positions, coeffs)
        pred_lmk_pos_ponderata = torch.sum(numeratore, dim=1)  # BATCH_SIZE x LMK x 3

        numeratore = torch.einsum("ijkm,ijk->ijkm", max_normals, coeffs)

        scale = data.scale[0]
        obj_path = data.obj_path[0]

        obj = o3d.io.read_triangle_mesh(obj_path)
        min_bound = [-1000, -1000, -75]
        max_bound = [1000, 1000, 1000]
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        obj = obj.crop(bounding_box)

        obj = obj.translate(-data.translation[0].cpu().numpy(), relative=True)
        obj = obj.scale(scale, center=[0, 0, 0])

        black_points = torch.tensor(np.asarray(obj.vertices), device=device, dtype=torch.float32)

        point_test = pred_lmk_pos_ponderata[0, :, :]
        t_dists = torch.cdist(black_points, point_test)

        max_positions = max_positions.squeeze(0)
        max_positions = max_positions.permute(1, 0, 2)

        dist_max = torch.max(torch.linalg.norm(max_positions[:, :, :] - point_test.unsqueeze(1).repeat(1, top_t, 1), dim=2), dim=1).values
        t_dists_flat = t_dists.flatten()
        dist_max_rep = dist_max.repeat(black_points.shape[0], 1).flatten()

        flat_bool = t_dists_flat < dist_max_rep * gamma
        bool = flat_bool.reshape(-1, n_landmarks)
        sum = torch.sum(bool, dim=1)
        t_indices = torch.argwhere(sum > 0).squeeze(1)
        V_tens = black_points[t_indices]

        max_positions = max_positions.to(device)
        max_positions_tens = max_positions[:, :, :]

        V_tens_rep = V_tens.unsqueeze(0).repeat(n_landmarks, top_t, 1)
        max_positions_tens_rep = max_positions_tens.repeat_interleave(V_tens.shape[0], dim=1)

        distances_tens = torch.linalg.norm(V_tens_rep - max_positions_tens_rep, dim=2)
        distances_tens = distances_tens.reshape(n_landmarks, top_t, -1).permute(0, 2, 1)
        observed_tens = max_values[0, :, :].permute(1, 0)
        observed_tens = observed_tens.unsqueeze(1)
        observed_tens = observed_tens.repeat(1, V_tens.shape[0], 1)

        hypothesis_tens = gaussian_distance(distances_tens, sigma)
        difference_tens = torch.linalg.norm(observed_tens - hypothesis_tens, dim=2)
        indices = torch.argmin(difference_tens, dim=1)
        best_v_tens = V_tens[indices]

        pond_dist = torch.linalg.norm(landmarks[:, :] - pred_lmk_pos_ponderata[0, :, :], dim=1) / scale
        ref_dist_smart = torch.linalg.norm(landmarks[:, :] - best_v_tens, dim=1) / scale

        ponds.append(pond_dist.cpu())
        smarts.append(ref_dist_smart.cpu())
        torch.cuda.empty_cache()

    if log:
        ponds = torch.tensor(np.array(ponds))
        pond_mean = ponds.mean(dim=0)
        pond_std = ponds.std(dim=0)
        smarts = torch.tensor(np.array(smarts))
        smart_mean = smarts.mean(dim=0)
        smart_std = smarts.std(dim=0)
        results = {"MSE_over_mesh_mean": smart_mean, "MSE_over_mesh_std": smart_std, "m_mean": pond_mean, "m_std": pond_std}
        torch.save(results, os.path.join(results_folder, "refine_results.pt"))
    return ponds, smarts
