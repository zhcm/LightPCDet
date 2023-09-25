import numpy as np
from pcdet.utils.common_utils import rotate_points_along_z


def random_flip_along_x(gt_boxes, points, return_flip=False):
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_flip=False):
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, return_rot=False):
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = rotate_points_along_z(points, np.array([noise_rotation]))
    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[:, 0:3], np.array([noise_rotation]))
    gt_boxes[:, 6] += noise_rotation
    if return_rot:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, return_scale=False):
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if return_scale:
        return gt_boxes, points, noise_scale
    return gt_boxes, points
