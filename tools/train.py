import sys
sys.path.insert(0, './')
import argparse
import os
import tqdm
import datetime
import torch
import torch.distributed as dist

from easydict import EasyDict
from tensorboardX import SummaryWriter
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network

from train_utils.train_utils import train_one_epoch
from train_utils.optimization import build_optimizer, build_scheduler


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # extra save path
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    # parameters
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    # interval
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='number of checkpoints to be evaluated')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    args.root_dir = '/remote-home/chenming.zhang/code/LightPCDet/output'
    args.exp_group_path = os.path.join(cfgs.DATA_CONFIG.DATASET, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if args.fix_random_seed:
        common_utils.set_random_seed(666 + local_rank)

    # savedir
    args.output_dir = os.path.join(args.root_dir, args.exp_group_path, args.tag, args.extra_tag)
    ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir) and local_rank == 0:
        os.makedirs(ckpt_dir)
    if local_rank == 0:
        common_utils.backup_source_code(os.path.join(args.output_dir, 'code'))
    if args.dist_mode:
        dist.barrier()

    # log
    log_file = os.path.join(args.output_dir, 'train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if local_rank == 0 else None

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)
    if local_rank == 0:
        os.system('cp %s %s' % (args.cfg_file, args.output_dir))

    # dataset
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfgs.DATA_CONFIG,
        det_class_names=cfgs.DET_CLASS_NAMES,
        batch_size=cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        dist=args.dist_mode,
        workers=args.workers,
        logger=logger,
        mode='training',
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfgs.MODEL, det_class_names=cfgs.DET_CLASS_NAMES)
    if cfgs.OPTIMIZATION.SYNC_BN and args.dist_mode:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info('Convert batch norm to sync batch norm')
    model.cuda()

    optimizer = build_optimizer(model, cfgs.OPTIMIZATION)

    last_epoch = -1
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfgs.OPTIMIZATION.NUM_EPOCHS,
        last_epoch=last_epoch, optim_cfg=cfgs.OPTIMIZATION)

    total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
    tbar = tqdm.trange(last_epoch + 1, total_epochs, desc='epochs', dynamic_ncols=True, leave=(local_rank == 0))
    if cfgs.OPTIMIZATION.MERGE_ALL_ITERS_TO_ONE_EPOCH:
        total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
    else:
        total_it_each_epoch = len(train_loader)

    accumulated_iter = 0
    dataloader_iter = iter(train_loader)  # 如果merge那只需要初始化一次
    for cur_epoch in tbar:
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)
        if lr_warmup_scheduler is not None and cur_epoch < cfgs.OPTIMIZATION.WARMUP_EPOCH:
            cur_scheduler = lr_warmup_scheduler
        else:
            cur_scheduler = lr_scheduler
        accumulated_iter = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            dataloader_iter=dataloader_iter,
            lr_scheduler=cur_scheduler,
            cur_epoch=cur_epoch,
            total_it_each_epoch=total_it_each_epoch,
            accumulated_iter=accumulated_iter,
            merge_all_iters_to_one_epoch=cfgs.OPTIMIZATION.MERGE_ALL_ITERS_TO_ONE_EPOCH)


if __name__ == '__main__':
    main()
