import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums)
        c_in_list = [input_channels] + num_filters[:-1]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx],
                          kernel_size=3, stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=num_filters[idx], out_channels=num_upsample_filters[idx],
                                   kernel_size=upsample_strides[idx], stride=upsample_strides[idx], bias=False),
                nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))

        channel_out = sum(num_upsample_filters)
        self.num_bev_features = channel_out

    def forward(self, data_dict):
        # spatial_features: [bz, 64, ny, nx]
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ups.append(self.deblocks[i](x))

        x = torch.cat(ups, dim=1)
        data_dict['spatial_features_2d'] = x
        return data_dict
