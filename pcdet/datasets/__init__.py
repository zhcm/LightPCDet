from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pcdet.utils import common_utils
from .openpit_poly_rs32.poly import PolyDataset

__all__ = {
    'PolyDataset': PolyDataset
}


def build_dataloader(dataset_cfg, det_class_names, batch_size,
                     dist, workers=4, logger=None, mode='training', seed=None):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        det_class_names=det_class_names,
        mode=mode,
        logger=logger)

    shuffle = True if mode == 'training' else False
    if dist:
        # DistributedSampler的drop_last参数表示数据不能均分到几张卡上的时候是丢弃还是补上
        # 在分配不同卡数据的时候，shuffle需要打乱顺序，使用的随机数种子是(seed+epoch)
        # seed和外面的设置无关=0，epoch通过训练中sampler.set_epoch设置，保证每个epoch分配给每个卡的数据不同
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        # sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        collate_fn=dataset.collate_batch,  # 默认使用default_collate，把batch_sampler取出的indexes组成batch的过程
        shuffle=(sampler is None) and (mode == 'training'),  # 没指定sampler的时候用于选择使用RandomSampler还是SequentialSampler
        drop_last=False,  # batch_sampler的参数，分batch时候如果不能均分要不要丢弃
        sampler=sampler,
        worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)  # 每个worker中的随机数种子
        # 解释1:
        # ddp模式下，每个卡上的进程都有workers个子进程，虽然每个卡设置了random_seed(666 + local_rank)，
        # 但是卡上的workers个子进程会继承相同的随机数，所以要额外设置
        # 解释2:
        # 通过阅读源码，发现在_worker_loop中有seed = base_seed + worker_id，所以解释1不正确，worker_init_fn也不需要指定
        # 为了保持一致，此处保留
    )

    return dataset, dataloader, sampler
