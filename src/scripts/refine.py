import os
import torch
from torch_cluster import knn_graph
from src.utils.utils import NormalizeScale
import potpourri3d as pp3d
import open3d as o3d

import numpy as np
import scipy
from tqdm.auto import tqdm
import time


# TODO: Optimize .pt creation in order to not apply the Rigid Transformation to each mesh again
@torch.no_grad()
def refine(model, refine_loader, results_folder, top_t=7, device="cuda", K_KNN=32, n_resampling_points=32768, log=True):
    model.eval()

    total_m = []
    total_loss = []

    pbar = tqdm(enumerate(refine_loader), total=len(refine_loader))
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

        global_edges[:, :] = torch.cartesian_prod(super_indexes[0, :], super_indexes[0, :]).T

        x = torch.cat([pos, normal], dim=-1)
        out, _ = model(x, global_edges, local_edges)

        interpr = out.reshape((batch_size, int(out.shape[0] / batch_size), out.shape[1]))

        max_values, max_indices = torch.topk(interpr, top_t, dim=1)

        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)

        max_values, max_indices = torch.topk(interpr, top_t, dim=1)
        max_indices = max_indices.repeat_interleave(3, dim=2).reshape(batch_size, -1, 3)
        max_index = torch.argmax(interpr, dim=1)
        max_index = max_index.repeat_interleave(3, dim=1).view(batch_size, -1, 3)

        pos = pos.reshape(batch_size, -1, 3)
        max_positions = pos.gather(1, max_indices).reshape(batch_size, top_t, -1, 3)

        coeffs = torch.softmax(max_values, dim=1)
        numeratore = torch.einsum("ijkm,ijk->ijkm", max_positions, coeffs)
        pred_lmk_pos_ponderata = torch.sum(numeratore, dim=1)

        scale = data.scale[0]
        obj_path = data.obj_path[0]
        obj = o3d.io.read_triangle_mesh(obj_path)
        min_bound = [-1000, -1000, -75]
        max_bound = [1000, 1000, 1000]
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        obj = obj.crop(bounding_box)

        pcd = obj.sample_points_poisson_disk(n_resampling_points)
        points_of_pcd = np.asarray(pcd.points)
        center = np.mean(points_of_pcd, axis=0)

        center_of_mass = obj.get_center()
        center_of_mass = center
        obj = obj.translate(-center_of_mass, relative=True)
        obj = obj.scale(scale, center=[0, 0, 0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = obj.vertices

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        lista_pond = []
        lm_losses = []

        for l in range(0, 68):
            point_test = pred_lmk_pos_ponderata[0, l, :].cpu().numpy()
            max_positions = max_positions.cpu()
            dist_max = torch.max(torch.linalg.norm(max_positions[0, :, l, :] - torch.tensor(point_test).repeat(top_t, 1), dim=1))
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point_test, dist_max * 3)
            np_triangles = np.asarray(obj.triangles)
            triangles = np_triangles[np.any(np.isin(np_triangles, idx), axis=1)]

            np_obj_points = np.asarray(obj.vertices)

            unique_indx_points = sorted(list(set(triangles.flatten())))
            points_tmp = np_obj_points[unique_indx_points]

            values = np.arange(len(unique_indx_points))
            dictionary = dict(zip(unique_indx_points, values))

            lista_finale = []
            for i, indx in enumerate(triangles.flatten()):
                lista_finale.append(dictionary[indx])

            F = np.asarray(lista_finale).reshape(-1, 3)
            V = np.asarray(points_tmp)

            query_points_on_patch = torch.argmin(torch.cdist(torch.tensor(points_tmp, dtype=torch.float32), max_positions[0, :, l, :]), dim=0)
            solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
            landmarks = landmarks.cpu()
            distances = []
            for query in query_points_on_patch:
                dist = solver.compute_distance(query)
                distances.append(dist)
            distances = np.asarray(distances)
            for i in range(top_t):
                distances[i] *= coeffs[0, i, l].item()
            scipy.special.softmax(distances.sum(axis=0))
            composizione = scipy.special.softmax(distances.sum(axis=0)) + scipy.special.softmax(distances.std(axis=0))

            l_hat = torch.tensor(V[composizione.argmin()])

            pred_lmk_pos_ponderata = pred_lmk_pos_ponderata.cpu()
            pond_dist = torch.linalg.norm(landmarks[l, :] - pred_lmk_pos_ponderata[0, l, :]) / scale
            pond_dist = pond_dist.item()
            lista_pond.append(pond_dist)
            loss = torch.linalg.norm(landmarks[l, :] - l_hat) / scale
            lm_losses.append(loss.item())

        total_m.append(np.asarray(lista_pond))
        total_loss.append(np.asarray(lm_losses))

    if log:
        ponds = torch.tensor(np.array(total_m))

        comp_mean = np.asarray(total_loss).mean(axis=0)
        comp_std = np.asarray(total_loss).std(axis=0)

        pond_mean = ponds.mean(dim=0)
        pond_std = ponds.std(dim=0)

        results = {"comp_mean": comp_mean, "comp_std": comp_std, "pond_mean": pond_mean, "pond_std": pond_std}
        torch.save(results, os.path.join(results_folder, "refine_results.pt"))
    return total_m, total_loss
