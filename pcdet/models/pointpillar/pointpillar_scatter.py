import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict):
        """
        pillar_features: [N, 64]
        voxel_coords: [N, 4]  [bz_index, z_index, y_index, x_index]
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1  # 0维保存的是属于batch中的哪个样本
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(self.num_bev_features, self.nz * self.nx * self.ny,
                                          dtype=pillar_features.dtype, device=pillar_features.device)
            #  self.nz * self.nx * self.ny 表示点云空间中 voxel的个数
            batch_mask = coords[:, 0] == batch_idx  # [N, ]
            this_coords = coords[batch_mask, :]  # [n, 4]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # [n, ]
            # 更准确的写法应该是
            # indices = this_coords[:, 1] * self.nx * self.ny  + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 表示一维下voxel对应具体的位置
            # 因为在z轴上只有1个voxel，this_coords[:, 1]永远==0
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]  # [n, 64]
            pillars = pillars.t()  # [64, n]
            spatial_feature[:, indices] = pillars  # [64, nz*nx*ny]
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)  # [bz, 64, nz*nx*ny]
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz,
                                                             self.ny, self.nx)
        # [bz, 64*1, ny, nx]  [N, C, H, W]
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
