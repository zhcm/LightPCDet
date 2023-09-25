# @Time    : 2023/9/22 17:07
# @Author  : zhangchenming
from .pointpillar.pointpillar import PointPillar

__all__ = {
    'PointPillar': PointPillar,
}

def build_network(model_cfg, det_class_names):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, det_class_names=det_class_names)

    return model