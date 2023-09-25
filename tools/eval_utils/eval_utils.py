import pickle
import numpy as np


def eval_one_epoch( model, dataloader, result_dir=None):
    det_annos = []

    for i, batch_dict in enumerate(dataloader):
        # pred_dict list, len=batch_size
        # [{'pred_boxes': tensor[N,7], 'pred_scores': tensor[N,1], 'pred_slabels': tensor[N,1]}, ...]
        pred_dicts = model(batch_dict)

        # annos list, len=batch_size
        # [{'name': numpy[N,], 'score': numpy[N,], 'boxes_lidar': numpy[N,7], 'pred_labels': numpy[N,], 'frame_id': str}, ......]
        # annos = dataset.generate_prediction_dicts(
        #     batch_dict, pred_dicts, class_names,
        #     output_path=final_output_dir if args.save_to_file else None
        # )
        # det_annos.extend(annos)

    # 记录所有的预测框
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # 计算mAP
    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    # )


def evaluation(det_annos, gt_annos):
    """
    det_annos: [{'name': numpy[N,], 'score': numpy[N,], 'boxes_lidar': numpy[N,7], 'pred_labels': numpy[N,], 'frame_id': str}, ......]
    gt_annos: [{'name': numpy[N,], 'gt_boxes_lidar': numpy[N,7], 'num_points_in_gt': numpy[N,]}, ......]
    (x,y,z,l,w,h,ry)
    """
    min_overlaps = [0.7, 0.5]


def calculate_iou(gt_annos, dt_annos, metric):
    dt_num_each_sample = np.stack([len(a["name"]) for a in dt_annos], 0)
    gt_num_each_sample = np.stack([len(a["name"]) for a in gt_annos], 0)

    num_examples = len(gt_annos)
    parted_overlaps = []

    # bev
    # gt_boxes: [N, 5] (x, y, l, w, ry), N是所有sample中所有框的总数，很大
    # dt_boxes: [M, 5] (x, y, l, w, ry), M是所有sample中所有框的总数，很大
    dt_boxes = np.concatenate([a["boxes_lidar"][:, [0, 1, 3, 4, 6]] for a in dt_annos], axis=0)
    gt_boxes = np.concatenate([a["gt_boxes_lidar"][:, [0, 1, 3, 4, 6]] for a in dt_annos], axis=0)

    overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)

    # 3d
    overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)

    parted_overlaps.append(overlap_part)


    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num




if __name__ == '__main__':
    pass
