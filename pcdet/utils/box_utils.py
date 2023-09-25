from pcdet.utils import common_utils


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        # [前方左下点, 前方右下点, 后方右下点, 后方左下点, 前方左上点, 前方右上点, 后方右上点, 后方左上点]
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1])) / 2

    # rots是以中心点为原点的旋转角度，所以现在把中心点为新的原点，(l,w,h) * template就可以看做是中心点为原点坐标系的八个点的坐标
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    # 对这八个点旋转之后求新坐标，主要公式为cos(A+B), sin(A+B), A是旋转角度，B是这个点和原点连线的角度
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    # 在加上中心点的坐标，即为原始点云坐标系中的坐标
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d

def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1, use_center_to_filter=True):
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    if use_center_to_filter:
        box_centers = boxes[:, 0:3]
        mask = ((box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])).all(axis=-1)
    else:
        corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
        corners = corners[:, :, 0:2]
        mask = ((corners >= limit_range[0:2]) & (corners <= limit_range[3:5])).all(axis=2)
        mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask