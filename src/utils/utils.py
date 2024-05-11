import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


# PREPROCESSING


@functional_transform("center_lmk")
class Center(BaseTransform):
    """
    Centers node positions :obj:`data.pos` around the origin
    (functional name: :obj:`center`).
    """

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, "pos") and hasattr(store, "landmarks"):
                mean = store.pos.mean(dim=-2, keepdim=True)
                store.pos = store.pos - mean
                store.landmarks = store.landmarks - mean
        return data


@functional_transform("normalize_scale_lmk")
class NormalizeScale(BaseTransform):
    """
    Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """

    def __init__(self):
        self.center = Center()

    def forward(self, data: Data) -> Data:
        data = self.center(data)
        scale = (1 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale
        data.landmarks = data.landmarks * scale
        data.scale = scale

        return data


@functional_transform("rotation_invariance")
class RotationInvariance(BaseTransform):
    r"""Rotates all points according to the eigenvectors of the point cloud
    (functional name: :obj:`rotation_invariance`).
    If the data additionally holds normals saved in :obj:`data.normal`, these
    will be rotated accordingly.

    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
        sort (bool, optional): If set to :obj:`True`, will sort eigenvectors
            according to their eigenvalues. (default: :obj:`False`)
    """

    def __init__(self, max_points: int = -1, sort: bool = False):
        self.max_points = max_points
        self.sort = sort

    def forward(self, data: Data) -> Data:
        pos = data.pos

        if self.max_points > 0 and pos.size(0) > self.max_points:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[: self.max_points]]

        pos = pos - pos.mean(dim=0, keepdim=True)
        C = torch.matmul(pos.t(), pos)
        e, v = torch.linalg.eig(C)  # v[:,j] is j-th eigenvector
        e, v = torch.view_as_real(e), v.real
        if self.sort:
            indices = e[:, 0].argsort(descending=True)
            v = v.t()[indices].t()
        centroid = torch.mean(data.pos, dim=0)
        mean_normal = torch.mean(data.normal, dim=0)
        distances = torch.cdist(data.pos, centroid.unsqueeze(0))
        farthest = torch.argmax(distances)

        direction = data.pos[farthest] - centroid
        x_angle = torch.acos(torch.dot(direction, v[:, 0]) / torch.norm(direction) * torch.norm(v[:, 0]))
        if x_angle > torch.tensor(torch.pi / 2):
            v[:, 0] = v[:, 0] * (torch.tensor(-1))
        z_angle = torch.acos(torch.dot(mean_normal, v[:, 2]) / torch.norm(mean_normal) * torch.norm(v[:, 2]))
        if z_angle > torch.tensor(torch.pi / 2):
            v[:, 2] = v[:, 2] * (torch.tensor(-1))
        v0_v2 = torch.cross(v[:, 0], v[:, 2])
        y_angle = torch.acos(torch.dot(v0_v2, v[:, 1]) / torch.norm(v0_v2) * torch.norm(v[:, 1]))
        if y_angle > torch.tensor(torch.pi / 2):
            v[:, 1] = v[:, 1] * (torch.tensor(-1))

        data.pos = torch.matmul(data.pos, v)
        if "normal" in data:
            data.normal = F.normalize(torch.matmul(data.normal, v))
        return data


def compute_points_normal(resampled_points: np.ndarray, mesh_points: np.ndarray, mesh_normals: np.ndarray) -> None:
    assert len(mesh_points) == len(mesh_normals)
    mesh_normals = torch.tensor(mesh_normals, dtype=torch.float32)
    normals = torch.zeros((len(resampled_points), 3))
    resampled_points_t = torch.tensor(resampled_points, dtype=torch.float32)
    points_t = torch.tensor(mesh_points, dtype=torch.float32)
    d = torch.cdist(resampled_points_t, points_t)
    indexes = torch.argmin(d, dim=1)
    normals = mesh_normals[indexes]
    return normals


# GROUND TRUTH CREATION


def gaussian_distance(distance, sigma):
    return torch.exp(-(distance**2) / (2 * sigma**2)).float()
