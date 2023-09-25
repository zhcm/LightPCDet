# @Time    : 2023/8/23 18:36
# @Author  : zhangchenming
import open3d
import numpy as np
import pickle
import math

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

    return vis


def open3d_draw(draw_origin, gt_boxes=None):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # 画坐标箭头，x:red, y:green, z:blue
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    points = np.fromfile('/remote-home/chenming.zhang/dataset/openpit_poly_rs32/velodyne/004158.bin', dtype=np.float32).reshape(-1, 4)
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    vis.run()
    vis.destroy_window()


def main():
    open3d_draw(draw_origin=True, gt_boxes=None)

    # with open('../resource/001082.txt', 'r') as f:
    #     lines = f.readlines()
    # gt_boxes = []
    # for line in lines:
    #     label = line.strip().split(' ')
    #     # [x, y, z, l, w, h, ry]
    #     gt_box = [float(label[11]), float(label[12]), float(label[13]),
    #               float(label[8]), float(label[9]), float(label[10]),
    #               float(label[14])]
    #     gt_boxes.append(gt_box)
    # gt_boxes = np.array(gt_boxes)
    # open3d_draw(draw_origin=True, gt_boxes=gt_boxes)

    # with open('../resource/kitti_infos_train.pkl', 'rb') as f:
    #     infos = pickle.load(f)
    # annos = infos[1]['annos']
    # gt_boxes_lidar = annos['boxes_lidar']
    # print(gt_boxes_lidar)
    # angle = gt_boxes_lidar[0][-1] * 180 / math.pi
    # print(angle)
    # open3d_draw(draw_origin=True, gt_boxes=gt_boxes_lidar)

    # with open('../resource/result.pkl', 'rb') as f:
    #     infos = pickle.load(f)
    # annos = infos[10]
    # gt_boxes_lidar = annos['boxes_lidar']
    # print(annos['frame_id'])
    # print(gt_boxes_lidar)
    # open3d_draw(draw_origin=True, gt_boxes=gt_boxes_lidar)

    # gt_boxes = np.array([[ 15.4089, -16.9471,  -2.8088,  10.2514,   4.7910,   4.8895,   1.9273],
    #     [ 56.0861,  33.5016,   0.2645,   8.9241,   4.2081,   4.1280,   5.2190],
    #     [  4.0252,  37.7109,  -0.8904,   9.2571,   4.3385,   4.8099,   4.7637],
    #     [ 20.0220,  20.2989,  -2.4022,  10.8038,   5.3443,   5.5265,   5.4197],
    #     [ 26.6512,  23.8119,  -0.8160,  10.2750,   5.0308,   4.7474,   5.0169],
    #     [ -1.0437,  13.4552,  -2.8031,  10.2351,   4.8701,   5.0336,   4.9757],
    #     [ -6.7410,  35.4758,  -4.0515,   9.9122,   4.5643,   4.5116,   5.2182],
    #     [ 46.3692,  28.7067,  -1.6999,   8.3298,   4.8037,   4.3400,   1.8384],
    #     [ 15.6227, -32.8051,  -2.8922,  10.7292,   5.0265,   5.6791,   5.6644],
    #     [-16.2562,   5.9286,  -3.4963,  11.1240,   5.1897,   5.8563,   4.7352],
    #     [ 43.1460,  20.4741,  -3.7018,   8.4759,   4.7380,   4.0320,   5.0111],
    #     [  8.3450,  -2.0230,  -3.6091,   9.0879,   4.7765,   5.1683,   6.4710]])
    # open3d_draw(draw_origin=True, gt_boxes=gt_boxes)


if __name__ == '__main__':
    main()
