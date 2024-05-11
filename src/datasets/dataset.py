from tqdm.auto import tqdm
import os
import os.path as osp

import numpy as np
import open3d as o3d
import openmesh

import torch
from torch_geometric.data import Data, Dataset

from src.utils.utils import compute_points_normal


class Faces3D(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, n_resampling_points=0):
        self.n_resampling_points = n_resampling_points
        super(Faces3D, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # Will check whether the file(s) in this list is already there in the "root" directory.
        raw_file_names = []
        for _, subject in enumerate(os.listdir(os.path.join(self.raw_dir))):
            if osp.isdir(os.path.join(self.raw_dir, subject)):
                for obj in os.listdir(os.path.join(self.raw_dir, subject)):
                    if obj.endswith(".obj"):
                        raw_file_names.append(osp.join(subject, obj))

        return raw_file_names

    @property
    def processed_file_names(self):
        # Will check whether the file(s) in this list is already there in the "root" directory.
        processed_paths = []
        for idx, _ in enumerate(self.raw_paths):
            processed_paths.append(osp.join(f"data_{idx}.pt"))
        return processed_paths

    def process(self):
        landmarks_indices = np.load(osp.join(self.root, "landmark_indices.npz"))["v10"]
        min_bound = [-1000, -1000, -75]
        max_bound = [1000, 1000, 1000]
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        total_landmarks = torch.zeros((len(self.raw_paths), 68, 3), dtype=torch.float32)
        total_scales = torch.zeros((len(self.raw_paths), 1), dtype=torch.float32)
        for i, path in enumerate(tqdm(self.raw_paths)):
            landmarks = np.zeros((68, 3))
            parts = path.split("/")[-1].split(".")[0].split("_")
            emotion = parts[1] if len(parts) == 2 else parts[1] + "_" + parts[2]
            try:
                om_mesh = openmesh.read_trimesh(path)
                landmarks = om_mesh.points()[landmarks_indices]
            except:
                print(f"Error in landmark extraction for image {path}")
                landmarks = None
            try:
                landmarks = torch.tensor(landmarks, dtype=torch.float32)
                if self.n_resampling_points > 0:
                    o3d_mesh = o3d.io.read_triangle_mesh(path)
                    o3d_mesh = o3d_mesh.crop(bounding_box)
                    o3d_mesh.compute_vertex_normals()
                    o3d_pcd = o3d_mesh.sample_points_poisson_disk(self.n_resampling_points)
                else:
                    o3d_pcd = o3d.geometry.PointCloud()
                    o3d_pcd.points = o3d.utility.Vector3dVector(om_mesh.points())

                np_o3d_points = np.asarray(o3d_pcd.points)
                normals = compute_points_normal(np_o3d_points, np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.vertex_normals))
                t_points = torch.tensor(np_o3d_points, dtype=torch.float32)
                data = Data(
                    pos=t_points,
                    normal=normals,
                    landmarks=landmarks,
                    emotion=emotion,
                    subject=path.split("/")[-2],
                    obj_path=path,
                )
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data.distances = torch.cdist(data.pos, data.landmarks)
                total_scales[i] = data.scale
                total_landmarks[i] = landmarks
                torch.save(data, osp.join(self.processed_dir, f"data_{i}.pt"))
            except Exception as e:
                print(f"Error in processing image {path}")
                print(e)

        mean_landmarks = torch.mean(total_landmarks, dim=0)
        mean_scale = torch.mean(total_scales)
        for i, data in enumerate(tqdm(self.processed_file_names)):
            pt_path = osp.join(self.processed_dir, data)
            data = torch.load(pt_path)
            cdist = torch.cdist(data.pos, mean_landmarks)
            indexes = torch.argmin(cdist, dim=0)
            data.super_indexes = indexes
            data.mean_scale = mean_scale
            torch.save(data, pt_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(osp.join(self.processed_dir, f"data_{index}.pt"))
        return data
