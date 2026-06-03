import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES


@LOSSES.register_module()
class ForestLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1):
        """
        Wraps CE + LovaszLoss with class remapping for forest segmentation.
        Original 7 classes -> Merged 6 classes:
          0=terrain, 1=foliage, 2=CWD, 3=woody(trunk+branch), 4=sng, 5=non-tree, 6=branch -> merged to woody(3)
        For 7-class evaluation, woody class IoU covers both trunk and branch.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.ce_weight = 1.0
        self.lovasz_weight = 1.0

    def forward(self, pred, target):
        merged_target = self._remap_target(target)
        valid_mask = merged_target != self.ignore_index
        merged_target_filtered = merged_target[valid_mask]
        pred_filtered = pred[valid_mask]

        num_classes = pred.size(1)
        ce_loss = F.cross_entropy(pred_filtered, merged_target_filtered, ignore_index=self.ignore_index)

        pred_prob = F.softmax(pred_filtered, dim=1)
        lovasz_loss = self._lovasz_softmax_flat(pred_prob, merged_target_filtered)

        return (ce_loss * self.ce_weight + lovasz_loss * self.lovasz_weight) * self.loss_weight

    def _remap_target(self, target):
        remap = torch.full_like(target, -1)
        remap[target == 0] = 0
        remap[target == 1] = 1
        remap[target == 2] = 2
        remap[(target == 3) | (target == 6)] = 3
        remap[target == 4] = 4
        remap[target == 5] = 5
        return remap

    def _lovasz_softmax_flat(self, probas, labels, classes="present"):
        C = probas.size(1)
        losses = []
        for c in labels.unique():
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            g = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, g))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)

    def _lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
