import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .. import utils

EPSILON = 1e-6

class sample_polar2cart(nn.Module):
    def __init__(
        self,
        z_max,
        z_min,
        cell_size,
    ):
        super().__init__()

        self.z_max = z_max
        self.z_min = z_min
        self.cell_size = cell_size

    def forward(self, features, calib, grid):

        # Normalise grid to [-1, 1]
        norm_grid = self.normalise_grid(grid, calib)

        bev_cart_feats = F.grid_sample(features, norm_grid, align_corners=True)
        return bev_cart_feats

    def normalise_grid(self, grid, calib):
        """
        :param grid: BEV grid in with coords range
        [grid_h1, grid_h2] and [-grid_w/2, grid_w/2]
        :param calib:
        :return:
        """

        f, cu = calib[:, 0, 0], calib[:, 0, 2]
        batch_size = len(calib)

        # Compute positive x dimension at z_max and z_min
        # Computed x dimension is half grid width
        x_zmax = self.z_max / f * cu
        x_zmin = self.z_min / f * cu

        # Compute normalising constant for each row along the z-axis
        sample_res_z = (self.z_max - self.z_min) / self.cell_size
        sample_res_x = grid.shape[2] / self.cell_size

        norm_z = (
            2
            * (grid[..., 1] - grid[..., 1].min())
            / (grid[..., 1].max() - grid[..., 1].min())
            - 1
        )

        norm_scale_x = torch.stack(
            [
                torch.linspace(float(x_zmin[i]), float(x_zmax[i]), int(sample_res_z))
                for i in range(batch_size)
            ]
        )

        grid_ones = torch.ones_like(grid)
        grid_ones[..., 0] *= norm_scale_x.view(batch_size, -1, 1).cuda()

        # Normalise grid to [-1, 1]
        norm_grid = grid / grid_ones

        norm_grid[..., 1] = norm_z

        return norm_grid