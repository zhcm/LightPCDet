# @Time    : 2023/9/22 09:31
# @Author  : zhangchenming
from pypcd import pypcd
import numpy as np
import os


def read_pcd(pcd_file_path):
    pcd = pypcd.PointCloud.from_path(pcd_file_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"])
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points


def main():
    count = 0
    root_dir = '/remote-home/chenming.zhang/dataset/custom'
    for each in os.listdir(os.path.join(root_dir, 'lidar')):
        np_points = read_pcd(os.path.join(root_dir, 'lidar', each))
        save_name = os.path.splitext(each)[0] + '.bin'
        np_points.tofile(os.path.join(root_dir, 'bin', save_name))
        count += 1
        print(count)


if __name__ == '__main__':
    main()
