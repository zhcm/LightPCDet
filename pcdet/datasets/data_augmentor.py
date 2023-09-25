import numpy as np
from pcdet.utils import common_utils
from pcdet.datasets.dataset_utils import augmentor_utils


class DataAugmentor(object):
    def __init__(self, augmentor_configs):
        self.data_augmentor_queue = []

        aug_config_list = augmentor_configs.AUG_CONFIG_LIST
        for cur_cfg in aug_config_list:
            cur_augmentor = globals()[cur_cfg.NAME](config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def forward(self, data_dict):
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict


class RandomWorldFlip:
    def __init__(self, config):
        self.config = config

    def __call__(self, data_dict):
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in self.config['ALONG_AXIS_LIST']:
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(gt_boxes, points, return_flip=True)
            data_dict['flip_%s' % cur_axis] = enable

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict


class RandomWorldRotation:
    def __init__(self, config):
        self.config = config

    def __call__(self, data_dict):
        rot_range = self.config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict


class RandomWorldScaling:
    def __init__(self, config):
        self.config = config

    def __call__(self, data_dict):
        gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], self.config['WORLD_SCALE_RANGE'], return_scale=True)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict


class LimitPeriod:
    def __init__(self, config):
        """
        offset=0.5, period=2pi的时候，会将弧度限制在-pi到pi之间
        """
        self.config = config

    def __call__(self, data_dict):
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi)
        return data_dict


class ShufflePoints:
    def __init__(self, config):
        self.config = config

    def __call__(self, data_dict):
        points = data_dict['points']
        shuffle_idx = np.random.permutation(points.shape[0])
        points = points[shuffle_idx]
        data_dict['points'] = points
        return data_dict