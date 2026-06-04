import torch
from utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            return pred
        loss = 0
        for c in self.criteria:
            l = c(pred, target)
            loss = loss + l.sum() if isinstance(l, torch.Tensor) and l.dim() > 0 else loss + l
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
