import yaml
import os
import shutil
import random
import logging
import numpy as np
import torch
from easydict import EasyDict

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_configs(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            logger.info('----------- %s -----------' % key)
            log_configs(cfgs[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def backup_source_code(backup_dir):
    # 子文件夹下的同名也会被忽略
    ignore_hidden = shutil.ignore_patterns(
        ".idea", ".git*", "*pycache*",
        "cfgs", "data", "output")

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree('.', backup_dir, ignore=ignore_hidden)


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    """
    offset = 0.5, period = 2pi
     -2pi <= val < -pi的时候， -1 <= val / period < -0.5, torch.floor(val / period + offset) = -1, ans = val + period, 0 <= ans < pi
    -pi <= val < 0的时候， -0.5 <= val / period < 0, torch.floor(val / period + offset) = 0, ans = val
    0 <= val < pi的时候， 0 <= val / period < 0.5, torch.floor(val / period + offset) = 0, ans = val
    pi <= val < 2pi的时候， 0.5 <= val / period < 1, torch.floor(val / period + offset) = 1, ans = val - period, -pi <= ans < 0
    2pi <= val < 3pi的时候, 1 <= val / period < 1.5, torch.floor(val / period + offset) = 1, ans = val - period, 0 <= ans < pi
    3pi <= val < 4pi的时候, 1.5 <= val / period < 2, torch.floor(val / period + offset) = 2, ans = val - 2period, -pi <= ans < 0
    """
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def rotate_points_along_z(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(size=[1])
    ones = angle.new_ones(size=[1])
    """
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    """
    rot_matrix = (torch.stack(tensors=(cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).
                  view(3, 3).float())

    points_rot = torch.matmul(points[:, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot