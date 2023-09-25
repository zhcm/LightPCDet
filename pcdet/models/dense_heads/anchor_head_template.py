import numpy as np
import torch
import torch.nn as nn
from .anchor_generator import AnchorGenerator

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        # 生成anchor
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors_list = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg,
            grid_size=grid_size).generate_anchors()
        self.anchors = [x.cuda() for x in anchors_list]

        # target_assigner = AxisAlignedTargetAssigner(
        #     model_cfg=self.model_cfg,
        #     class_names=self.class_names,
        #     box_coder=self.box_coder,
        #     match_height=anchor_target_cfg.MATCH_HEIGHT
        # )


    def forward(self, **kwargs):
        raise NotImplementedError
