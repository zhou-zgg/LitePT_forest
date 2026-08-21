import os
import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

from datasets.transform import Compose
from metrics.semantic import ConfusionMatrix

from torch_scatter import scatter_mean
from torch.nn.functional import normalize
from tqdm import tqdm

@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=True):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        # 圆柱分块评估：点数超过 val_crop_point_max 的文件按点密度自适应半径
        # 切成多个圆柱块全覆盖前向（与训练 CylinderCropCUDA 同分布），
        # softmax 概率累加合并，避免大文件整图前向 OOM。
        block_max_pts = getattr(self.trainer.cfg, "val_crop_point_max", None)
        for i, input_dict in enumerate(self.trainer.val_loader):
            output_dict = output = None
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            segment = input_dict["segment"]
            need_block = (
                block_max_pts is not None and segment.shape[0] > block_max_pts
            )
            if not need_block:
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)
                output = output_dict["seg_logits"]
                loss = output_dict["loss"]
                pred = output.max(1)[1]
                if "inverse" in input_dict.keys():
                    assert "origin_segment" in input_dict.keys()
                    pred = pred[input_dict["inverse"]]
                    segment = input_dict["origin_segment"]
            else:
                pred, loss = self._block_eval(input_dict, block_max_pts)
                if "inverse" in input_dict.keys():
                    assert "origin_segment" in input_dict.keys()
                    pred = pred[input_dict["inverse"]]
                    segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
            # 释放本次前向的 GPU 显存，避免连续评估多个大文件时显存累积 OOM
            del output_dict, output, pred, segment, loss
            torch.cuda.empty_cache()
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        total_pts = sum(target)
        tp = intersection.astype(np.float64)
        gt_count = target.astype(np.float64)
        fp = (union - target).astype(np.float64)
        fn = (target - intersection).astype(np.float64)
        tn = (total_pts - union).astype(np.float64)
        pred_count = tp + fp

        iou_class = tp / (tp + fp + fn + 1e-10)
        acc_class = tp / (gt_count + 1e-10)
        prec_class = tp / (pred_count + 1e-10)
        recall_class = acc_class.copy()
        f1_class = 2 * prec_class * recall_class / (prec_class + recall_class + 1e-10)
        spec_class = tn / (tn + fp + 1e-10)

        # 只对有 GT 点的类别求均值：被 ignore_classes 忽略的类（如 CWD，
        # 标签已映射为 -1）GT 为 0，不参与 mIoU 等均值，避免空类拖低指标。
        valid_cls = gt_count > 0

        m_iou = np.mean(iou_class[valid_cls]) if valid_cls.any() else 0.0
        m_acc = np.mean(acc_class[valid_cls]) if valid_cls.any() else 0.0
        all_acc = tp.sum() / (total_pts + 1e-10)
        m_prec = np.mean(prec_class[valid_cls]) if valid_cls.any() else 0.0
        m_recall = np.mean(recall_class[valid_cls]) if valid_cls.any() else 0.0
        m_f1 = np.mean(f1_class[valid_cls]) if valid_cls.any() else 0.0
        m_spec = np.mean(spec_class[valid_cls]) if valid_cls.any() else 0.0
        fw_iou = np.sum(gt_count / (total_pts + 1e-10) * iou_class)

        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        self.trainer.logger.info(
            "Val enhanced: mPrec/mRecall/mF1/mSpec/FWIoU {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                m_prec, m_recall, m_f1, m_spec, fw_iou
            )
        )
        names = self.trainer.cfg.data.names
        nc = self.trainer.cfg.data.num_classes
        header = f"{'Class':>16s} {'IoU':>7s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Spec':>7s} {'GT#':>10s} {'Pred#':>10s}"
        self.trainer.logger.info(header)
        for i in range(nc):
            self.trainer.logger.info(
                f"{names[i]:>16s} {iou_class[i]:>7.4f} {acc_class[i]:>7.4f} "
                f"{prec_class[i]:>7.4f} {recall_class[i]:>7.4f} "
                f"{f1_class[i]:>7.4f} {spec_class[i]:>7.4f} "
                f"{int(gt_count[i]):>10d} {int(pred_count[i]):>10d}"
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def _block_eval(self, input_dict, max_pts):
        """圆柱形空间分块全覆盖评估。

        与训练的 CylinderCropCUDA 同分布：按点密度自适应圆柱半径，圆柱中心
        以方形网格（步长 1.2r，保证相邻圆柱重叠、全覆盖）铺满 XY 范围，
        逐块前向，softmax 概率按点累加，重叠区取概率平均后 argmax。
        返回 (pred [N], loss tensor)。
        """
        import math

        coord = input_dict["coord"]  # GridSample 后的降采样点
        num_classes = self.trainer.cfg.data.num_classes
        n = coord.shape[0]
        xy = coord[:, :2]
        x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
        x_max, y_max = float(xy[:, 0].max()), float(xy[:, 1].max())

        # 自适应半径：目标点数 / 点密度 → 圆面积 → 半径
        area = max((x_max - x_min) * (y_max - y_min), 1e-6)
        density = n / area
        r = math.sqrt(max_pts / (math.pi * density))
        r = min(max(r, 5.0), 60.0)
        step = 1.2 * r

        centers = []
        cx = x_min - r
        while cx <= x_max + r:
            cy = y_min - r
            while cy <= y_max + r:
                centers.append((cx, cy))
                cy += step
            cx += step

        probs = torch.zeros(n, num_classes, device=coord.device)
        counts = torch.zeros(n, device=coord.device)
        loss_sum, loss_w = 0.0, 0
        hard_cap = int(1.5 * max_pts)  # 过密圆柱截断，截掉点由相邻重叠圆柱覆盖
        n_blocks = 0
        for cx, cy in centers:
            d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
            idx = torch.nonzero(d2 <= r * r, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            if idx.numel() > hard_cap:
                perm = torch.randperm(idx.numel(), device=idx.device)[:hard_cap]
                idx = idx[perm]
            block = {}
            for k in ("coord", "grid_coord", "segment", "feat"):
                if k in input_dict:
                    block[k] = input_dict[k][idx]
            # 单块即单 batch：显式给 batch 索引，Point 会自动生成对应 offset
            block["batch"] = torch.zeros(
                idx.numel(), dtype=torch.long, device=idx.device
            )
            # offset 是 [batch_size] 形状，不能按点索引，直接丢弃由 Point 重建
            block.pop("offset", None)
            # 圆柱平移到原点（grid_coord 取整数格偏移，保持网格对齐），
            # 贴近训练时 CenterShift 后的坐标分布
            if "grid_coord" in block:
                block["grid_coord"] = block["grid_coord"] - block[
                    "grid_coord"
                ].min(0).values
            with torch.no_grad():
                out = self.trainer.model(block)
            probs[idx] += torch.softmax(out["seg_logits"], dim=-1)
            counts[idx] += 1
            loss_sum += float(out["loss"].item()) * idx.numel()
            loss_w += idx.numel()
            n_blocks += 1
            del block, out
            torch.cuda.empty_cache()

        uncovered = int((counts == 0).sum().item())
        if uncovered > 0:
            self.trainer.logger.warning(
                "Block eval: {} points not covered, fallback to mini-batch forward".format(
                    uncovered
                )
            )
            # 兜底：密集区被截断漏掉的点，分小批直接前向
            rest_idx = torch.nonzero(counts == 0, as_tuple=False).squeeze(1)
            batch = 50000
            for s in range(0, rest_idx.numel(), batch):
                idx = rest_idx[s : s + batch]
                block = {}
                for k in ("coord", "grid_coord", "segment", "feat"):
                    if k in input_dict:
                        block[k] = input_dict[k][idx]
                block["batch"] = torch.zeros(
                    idx.numel(), dtype=torch.long, device=idx.device
                )
                if "grid_coord" in block:
                    block["grid_coord"] = block["grid_coord"] - block[
                        "grid_coord"
                    ].min(0).values
                with torch.no_grad():
                    out = self.trainer.model(block)
                probs[idx] += torch.softmax(out["seg_logits"], dim=-1)
                counts[idx] += 1
                loss_sum += float(out["loss"].item()) * idx.numel()
                loss_w += idx.numel()
                del block, out
                torch.cuda.empty_cache()
        pred = probs.argmax(1)
        pred[counts == 0] = 0
        loss = torch.tensor(loss_sum / max(loss_w, 1), device=coord.device)
        self.trainer.logger.info(
            "Block eval: {} pts, r={:.1f}m, {} blocks".format(n, r, n_blocks)
        )
        return pred, loss

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/mAP": all_ap,
                        "val/AP50": all_ap_50,
                        "val/AP25": all_ap_25,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
