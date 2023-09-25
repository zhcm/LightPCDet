# @Time    : 2023/9/24 14:05
# @Author  : zhangchenming
import torch
import torch.nn as nn
import numpy as np
from .pillar_vfe import PillarVFE
from .pointpillar_scatter import PointPillarScatter
from .base_bev_backbone import BaseBEVBackbone
from .anchor_head_single import AnchorHeadSingle

class PointPillar(nn.Module):
    def __init__(self, model_cfg, det_class_names):
        super().__init__()
        self.model_cfg = model_cfg
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.point_cloud_range = np.array(self.model_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.voxel_size = self.model_cfg.VOXEL_SIZE
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # vfe
        self.vfe = PillarVFE(model_cfg=self.model_cfg.VFE, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
        # map_to_bev
        self.map_to_bev_module = PointPillarScatter(model_cfg=self.model_cfg.MAP_TO_BEV, grid_size=self.grid_size)
        # 2d backbone
        self.backbone_2d = BaseBEVBackbone(model_cfg=self.model_cfg.BACKBONE_2D, input_channels=self.map_to_bev_module.num_bev_features)
        # head
        AnchorHeadSingle(model_cfg=self.model_cfg.DENSE_HEAD,
                         point_cloud_range=self.point_cloud_range,
                         grid_size=self.grid_size,
                         input_channels=self.backbone_2d.num_bev_features,
                         det_class_names=det_class_names)



    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
