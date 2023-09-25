# @Time    : 2023/9/7 16:50
# @Author  : zhangchenming
import os
import numpy as np

limit_range_min = [0, -40]
limit_range_max = [70, 40]


def get_bakground():
    index_id = '1609317398.740745000'

    lidar_file = '/remote-home/chenming.zhang/dataset/custom/bin/%s.bin' % index_id
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    mask = (points[:, 0] >= limit_range_min[0]) & (points[:, 0] <= limit_range_max[0]) \
           & (points[:, 1] >= limit_range_min[1]) & (points[:, 1] <= limit_range_max[1])
    points = points[mask]
    points.tofile('/remote-home/chenming.zhang/dataset/custom/bg.bin')


def remove_background():
    threshold = 0.5 ** 2
    root_path = '/remote-home/chenming.zhang/dataset/custom'

    # 读取背景点云
    bg_points = np.fromfile(os.path.join(root_path, 'bg.bin'), dtype=np.float32).reshape(-1, 4)

    for each in os.listdir(os.path.join(root_path, 'bin')):
        # 读取点云并过滤范围
        lidar_file = os.path.join(root_path, 'bin', each)
        pc_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        mask = (pc_points[:, 0] >= limit_range_min[0]) & (pc_points[:, 0] <= limit_range_max[0]) \
               & (pc_points[:, 1] >= limit_range_min[1]) & (pc_points[:, 1] <= limit_range_max[1])
        pc_points = pc_points[mask]

        # 计算距离
        dist_part1 = -2 * np.matmul(bg_points, pc_points.transpose(1, 0))
        dist_part2 = np.expand_dims(np.sum(bg_points ** 2, axis=-1), axis=-1)
        dist_part3 = np.expand_dims(np.sum(pc_points ** 2, axis=-1), axis=0)
        dist_final = dist_part1 + dist_part2 + dist_part3

        # 因为计算精度问题可能有负值
        # dist_final = torch.sqrt(dist_final)
        dist_final = np.min(dist_final, axis=0)
        mask = dist_final > threshold
        print(np.sum(mask))
        new_points = pc_points[mask]

        # 保存
        new_points.tofile(os.path.join(root_path, 'bin_nobg', each))


if __name__ == '__main__':
    remove_background()
