import torch
from torch import nn


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss from
    Wang X, Bo L, Fuxin L. Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression. ICCV2019.
    The following module is based on https://github.com/protossw512/AdaptiveWingLoss
    """

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        tmp = y.clone()
        y_gauss_multiplier = y.clone()
        tmp[tmp > 0.1] = 1
        tmp[tmp <= 0.1] = 0
        y_gauss_multiplier_1 = y_gauss_multiplier[delta_y < self.theta]
        y_gauss_multiplier_2 = y_gauss_multiplier[delta_y >= self.theta]
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y1_high = tmp[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        delta_y2_high = tmp[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.epsilon, self.alpha - y1))
        A = self.omega * (self.alpha - y2) * torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1) / (torch.pow(1 + (self.theta / self.epsilon), self.alpha - y2) / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        loss1 = loss1 * (delta_y1_high * 50 * y_gauss_multiplier_1 + 1)
        loss2 = loss2 * (delta_y2_high * 50 * y_gauss_multiplier_2 + 1)
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        distances = torch.linalg.norm(target - pred, dim=1)
        return distances.mean()
