import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def get_paddings_indicator(actual_num, max_num, axis=0):
    # actual_num [N, ]
    # max_num 32
    actual_num = torch.unsqueeze(actual_num, axis + 1) # [N, 1]
    max_num_shape = [1] * len(actual_num.shape)  # value=[1, 1]
    max_num_shape[axis + 1] = -1  # value=[1, -1]
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)  # [1, 32]
    # [[0,1,2,3,4,5,......,30,31]]
    paddings_indicator = actual_num.int() > max_num  # [N, 32]
    # 每个值，和0到32比较，会得到一个长度32的向量，小于这个值的为true，大于这个值的为false，对应了真实点的mask
    return paddings_indicator


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg.USE_NORM
        self.num_filters = self.model_cfg.NUM_FILTERS

        num_point_features = 4 + 6
        num_filters = [num_point_features] + list(self.num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict):
        """
        'voxels': [N, 32, 4]  点云特征为[x, y, z, 反射强度]， 32为一个 voxel 中的点云个数，有补0的点
        'voxel_num_points': [N,] 每个voxel中真实点的个数
        'voxel_coords': [N, 4]  voxel在真实空间格子中的位置，[bz_index, z_index, y_index, x_index]
        """
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        # voxel中真实点的平均位置(世界坐标)
        points_sum = voxel_features[:, :, :3].sum(dim=1, keepdim=True)  # [N, 1, 3]
        points_mean =  points_sum / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)  # [N, 1, 3]
        # 表示 voxel 中每个点相对于这个平均位置的偏移量
        f_cluster = voxel_features[:, :, :3] - points_mean  # [N, 32, 3]

        f_center = torch.zeros_like(voxel_features[:, :, :3])  # [N, 32, 3]
        # 每个 voxel 正方体的中心位置（世界坐标）
        voxel_center_x = coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        voxel_center_y = coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        voxel_center_z = coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        # 表示 voxel 中每个点相对于这个中心位置的偏移量
        f_center[:, :, 0] = voxel_features[:, :, 0] - voxel_center_x
        f_center[:, :, 1] = voxel_features[:, :, 1] - voxel_center_y
        f_center[:, :, 2] = voxel_features[:, :, 2] - voxel_center_z

        features = [voxel_features, f_cluster, f_center]
        features = torch.cat(features, dim=-1)  # [N, 32, 10]

        # voxel_features [N, 32, 4], 32为每个voxel中point的个数，通常不够32个，用0来补位
        # 用0补位的点，在计算f_cluster和f_center的过程中结果不为0，相当于给这些点引入了有激活的feature
        # 因此需要get_paddings_indicator把他们重新变为0
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict