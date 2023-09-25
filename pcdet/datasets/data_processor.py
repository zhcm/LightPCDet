import numpy as np
from pcdet.utils import box_utils
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import cumm.tensorview as tv

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, mode):
        self.data_processor_queue = []

        for cur_cfg in processor_configs:
            cur_processor = globals()[cur_cfg.NAME](config=cur_cfg, point_cloud_range=point_cloud_range, mode=mode)
            self.data_processor_queue.append(cur_processor)

    def forward(self, data_dict):
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        return data_dict

class MaskPointsAndBoxesOutsideRange:
    def __init__(self, config, point_cloud_range, mode):
        self.config = config
        self.point_cloud_range = point_cloud_range
        self.mode = mode

    def __call__(self, data_dict):
        mask = ((data_dict['points'][:, 0] >= self.point_cloud_range[0])
                & (data_dict['points'][:, 0] <= self.point_cloud_range[3])
                & (data_dict['points'][:, 1] >= self.point_cloud_range[1])
                & (data_dict['points'][:, 1] <= self.point_cloud_range[4]))
        data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and self.config.REMOVE_OUTSIDE_BOXES:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range,
                min_num_corners=self.config.get('min_num_corners', 1),
                use_center_to_filter=self.config.get('USE_CENTER_TO_FILTER', True))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

class TransformPoints2Voxels:
    def __init__(self, config, point_cloud_range, mode):
        self.config = config
        self.point_cloud_range = point_cloud_range
        self.mode = mode
        self.spconv_ver = 2
        self.voxel_size = config.VOXEL_SIZE

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.voxel_generator = VoxelGenerator(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=4,
            max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
            max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
        )

    def __call__(self, data_dict):
        points = data_dict['points']
        voxel_output = self.voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict
