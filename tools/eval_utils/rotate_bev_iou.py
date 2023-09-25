# @Time    : 2023/9/19 15:51
# @Author  : zhangchenming
import math
import numpy as np


def rbbox_to_corners(rbbox):
    """
    args:
        rbbox: [center_x, center_y, w, h, angle]
    return:
        corners: [x1, y1, x2, y2, .....]
    """
    cx, cy, x_d, y_d, angle = rbbox
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    corners_x = [-x_d / 2, -x_d / 2, x_d / 2, x_d / 2]
    corners_y = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
    corners = [0] * 8
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] - a_sin * corners_y[i] + cx
        corners[2 * i + 1] = a_sin * corners_x[i] + a_cos * corners_y[i] + cy

    return np.array(corners, dtype=np.float32)


def point_in_quadrilateral(pt_x, pt_y, corners):
    """
    判断一个点是不是在四个角点围成四边形的内部
    """
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap >= 0 and adad >= adap >= 0


def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.array((2,), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = np.array((2,), dtype=np.float32)
    B = np.array((2,), dtype=np.float32)
    C = np.array((2,), dtype=np.float32)
    D = np.array((2,), dtype=np.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.array((2,), dtype=np.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.array((2,), dtype=np.float32)
        vs = np.array((16,), dtype=np.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty

def inter(rbbox1, rbbox2):
    """Compute intersection of two rotated boxes.
    Args:
        rbbox1 (np.ndarray, shape=[5]): Rotated 2d box.
        rbbox2 (np.ndarray, shape=[5]): Rotated 2d box.
    Returns:
        float: Intersection of two rotated boxes.
    """
    intersection_corners = np.array((16,), dtype=np.float32)

    corners1  = rbbox_to_corners(rbbox1)
    corners2  = rbbox_to_corners(rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


def rotate_iou(rbox1, rbox2, criterion=-1):
    """
    :param rbox1: [x, y, l, w, r]
    :param rbox2:
    :param criterion:
    :return:
    """
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


if __name__ == '__main__':
    rbbox_ = [0, 0, 2, 4, math.pi / 2]
    print(rbbox_to_corners(rbbox_))