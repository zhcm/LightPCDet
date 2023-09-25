import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config, grid_size):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.grid_size = grid_size

    def generate_anchors(self):
        all_anchors = []
        for config in self.anchor_generator_cfg:
            anchor_size = config['anchor_sizes']
            anchor_rotation = config['anchor_rotations']
            anchor_height = config['anchor_bottom_heights']
            align_center = config.get('align_center', False)
            feature_map_size = self.grid_size[:2] // config['feature_map_stride']

            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / feature_map_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / feature_map_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (feature_map_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (feature_map_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            # [x, y, z, 3]：x：在x方向上的点数，y，在y方向上的点数，z，高度的点数，3，每个点的坐标
            # 此时还没有加入anchor size，表示的只是 多少个anchor的中心点
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            # [x_grid, y_grid, height, num_anchor_size, 6]
            # x,y格子数量，由点云范围，体素大小，下采样尺度决定
            # height，设置的高度数量
            # num_anchor_size：设置的几个尺度
            # 6个值，表示x,y,z,l,w,h,坐标，长宽高
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            # [x_grid, y_grid, num_height, num_anchor_size, num_rot, 7]
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            # [num_height, y_grid, x_grid, num_anchor_size, num_rot, 7]
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            # z + h/2,配置文件中的height指的是框的底部高度
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors
