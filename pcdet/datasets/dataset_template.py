from collections import defaultdict
import numpy as np
import torch.utils.data as torch_data
from pcdet.datasets.data_augmentor import DataAugmentor
from pcdet.datasets.data_processor import DataProcessor


class DatasetTemplate(torch_data.Dataset):
    def __init__(self,
                 dataset_cfg=None,
                 mode='training',
                 logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.mode = mode
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.logger = logger
        self.root_path = self.dataset_cfg.DATA_PATH

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.data_augmentor = DataAugmentor(self.dataset_cfg.DATA_AUGMENTOR) if (self.mode == 'training') else None
        self.data_processor = DataProcessor(self.dataset_cfg.DATA_PROCESSOR,
                                            point_cloud_range=self.point_cloud_range,
                                            mode=mode)

    def prepare_data(self, data_dict):
        if self.mode == 'training':
            data_dict = self.data_augmentor.forward(data_dict=data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.mode == 'training' and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        return data_dict

    def merge_all_iters_to_one_epoch(self, merge=True, total_epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = total_epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        batch_list: [idx1, idx2, idx3, idx4]
        idx:
        {'points': array(n, 4),
         'voxel': array(n, 32, 4),  [x, y, z, intensity]
         'voxel_coords': array(n, 3),  [nz, ny, nx]
         'voxel_num_points': array(n,),
         'gt_boxes': array(n, 8)}

        data_dict:
        {'points': [array(n1, 4), array(n2, 4), array(n3, 4), array(n4, 4)]
         'voxel': [array(n1, 32, 4), array(n2, 32, 4), array(n3, 32, 4), array(n4, 32, 4)]
         'voxel_coords': [array(n1, 3), array(n2, 3), array(n3, 3), array(n4, 3)]
         'voxel_num_points': [array(n,), array(n,), array(n,), array(n,)]
         'gt_boxes': [array(n, 8), array(n, 8), array(n, 8), array(n, 8)]
         }

        ret:
        {'points': array(n1+n2+n3+n4, 5),  # 在第一个位置上补了index
         'voxel': array(n1+n2+n3+n4, 32, 4),
         'voxel_coords': array(n1+n2+n3+n4, 4),  # 在第一个位置上补了index
         'voxel_num_points': array(n1+n2+n3+n4, )
         'gt_boxes': array(batch_size, max(n1, n2, n3, n4), 8), # 统一成了一样的维度然后组batch
        }
        """
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)

            elif key in ['points', 'voxel_coords']: # 在数据特征上增加一维，表示属于哪个index
                xs = []
                for i, x in enumerate(val):
                    # 在每个坐标前面加上序号 shape (N, 4) -> (N, 5)  [20, 30, 40, 0.4] -> [i, 20, 30, 40, 0.4]
                    # np.pad 每个轴要填充的数据的数目((0, 0), (1, 0)) 表示0轴都填充0个，1轴在前面填充一个数
                    x_pad = np.pad(x, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    xs.append(x_pad)
                ret[key] = np.concatenate(xs, axis=0)

            elif key in ['gt_boxes']: # 补零然后concat, [bz, max_num, 8]
                max_gt = max([x.shape[0] for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].shape[0], :] = val[k]
                ret[key] = batch_gt_boxes3d

            else:  # frame_id, flip_x, noise_rot, noise_scale
                ret[key] = np.stack(val, axis=0)

        ret['batch_size'] = batch_size
        return ret
