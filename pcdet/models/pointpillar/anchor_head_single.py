import numpy as np
import torch.nn as nn

from pcdet.models.dense_heads.anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, point_cloud_range, grid_size, input_channels, det_class_names):
        super().__init__(model_cfg=model_cfg,
                         point_cloud_range=point_cloud_range,
                         grid_size=grid_size)

        self.det_class_names=det_class_names

        num_anchors_per_location_list = []
        for config in self.model_cfg.ANCHOR_GENERATOR_CONFIG:
            anchor_sizes = config['anchor_sizes']
            anchor_rotations = config['anchor_rotations']
            anchor_bottom_heights = config['anchor_bottom_heights']
            num_anchors_per_location = len(anchor_sizes) * len(anchor_rotations) * len(anchor_bottom_heights)
            num_anchors_per_location_list.append(num_anchors_per_location)
        self.num_anchors_per_location = sum(num_anchors_per_location_list)

        self.conv_cls = nn.Conv2d(in_channels=input_channels,
                                  out_channels=self.num_anchors_per_location * len(self.det_class_names),
                                  kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels=input_channels,
                                  out_channels=self.num_anchors_per_location * 7,
                                  kernel_size=1)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(in_channels=input_channels,
                                          out_channels=self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                                          kernel_size=1)
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
