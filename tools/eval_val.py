import os, sys, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engines.defaults import default_argument_parser, default_config_parser, default_setup
from engines.launch import launch
from datasets.builder import build_dataset
from datasets.utils import collate_fn
from models.builder import build_model
from utils.logger import get_root_logger


def main_worker(cfg):
    cfg = default_setup(cfg)
    logger = get_root_logger()

    logger.info(f"=> Loading config ...")
    model = build_model(cfg.model).cuda()
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters())}")

    if cfg.weight:
        ckpt = torch.load(cfg.weight, map_location="cpu", weights_only=False)
        sd = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            sd[k[7:] if k.startswith("module.") else k] = v
        model.load_state_dict(sd, strict=True)
        logger.info(f"=> Loaded weight '{cfg.weight}' (epoch {ckpt['epoch']})")

    model.eval()
    val_dataset = build_dataset(cfg.data.val)
    logger.info(f"Val set: {len(val_dataset)} samples")

    num_classes = cfg.data.num_classes
    ignore_index = cfg.data.ignore_index
    names = cfg.data.names
    save_dir = os.path.join(cfg.save_path, "result")
    os.makedirs(save_dir, exist_ok=True)

    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_sample = {}

    for idx in range(len(val_dataset)):
        data_dict = val_dataset[idx]
        data_name = val_dataset.get_data_name(idx)
        keys_to_pop = ["segment", "origin_segment", "inverse"]
        segment = data_dict.pop("segment") if "segment" in data_dict else None
        origin_segment = data_dict.pop("origin_segment") if "origin_segment" in data_dict else None
        inverse = data_dict.pop("inverse") if "inverse" in data_dict else None

        if isinstance(segment, torch.Tensor): segment = segment.cpu().numpy()
        if isinstance(origin_segment, torch.Tensor): origin_segment = origin_segment.cpu().numpy()
        if isinstance(inverse, torch.Tensor): inverse = inverse.cpu().numpy()

        input_dict = collate_fn([data_dict])
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input_dict)["seg_logits"]
        pred = output.max(1)[1].cpu().numpy()

        if inverse is not None:
            pred = pred[inverse]
        gt = origin_segment if origin_segment is not None else segment

        np.save(os.path.join(save_dir, f"{data_name}_pred.npy"), pred)

        mask = gt != ignore_index
        pred_f, gt_f = pred[mask], gt[mask]
        for t, p in zip(gt_f, pred_f):
            total_confusion[t, p] += 1

        sample_iou = []
        for c in range(num_classes):
            tp = total_confusion[c, c]
            union = total_confusion[c].sum() + total_confusion[:, c].sum() - tp
            sample_iou.append(tp / max(union, 1))

        valid = [i for i in range(num_classes) if total_confusion[i].sum() > 0 or total_confusion[:, i].sum() > 0]
        miou = np.mean([sample_iou[c] for c in valid]) if valid else 0.0
        logger.info(f"[{idx+1}/{len(val_dataset)}] {data_name} running mIoU={miou:.4f}")
        per_sample[data_name] = {names[c]: round(sample_iou[c], 4) for c in range(num_classes)}

    tp_per_class = np.diag(total_confusion).astype(np.float64)
    gt_count = total_confusion.sum(axis=1).astype(np.float64)
    pred_count = total_confusion.sum(axis=0).astype(np.float64)
    total_points = total_confusion.sum()

    iou_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    specificity_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        tp = tp_per_class[c]
        union = gt_count[c] + pred_count[c] - tp
        iou_per_class[c] = tp / max(union, 1)
        precision_per_class[c] = tp / max(pred_count[c], 1)
        recall_per_class[c] = tp / max(gt_count[c], 1)
        f1_per_class[c] = (2 * precision_per_class[c] * recall_per_class[c]) / max(
            precision_per_class[c] + recall_per_class[c], 1e-10
        )
        tn = total_points - gt_count[c] - pred_count[c] + tp
        fp = pred_count[c] - tp
        fn = gt_count[c] - tp
        specificity_per_class[c] = tn / max(tn + fp, 1)

    acc_per_class = tp_per_class / np.maximum(gt_count, 1)
    miou = np.mean(iou_per_class)
    macc = np.mean(acc_per_class)
    allacc = tp_per_class.sum() / max(total_points, 1)
    mprecision = np.mean(precision_per_class)
    mrecall = np.mean(recall_per_class)
    mf1 = np.mean(f1_per_class)
    mspecificity = np.mean(specificity_per_class)
    fwiou = np.sum(gt_count / max(total_points, 1) * iou_per_class)

    logger.info(f"\n{'='*80}")
    logger.info(f"Overall Metrics:")
    logger.info(f"  mIoU={miou:.4f}  mAcc={macc:.4f}  allAcc={allacc:.4f}")
    logger.info(f"  mPrecision={mprecision:.4f}  mRecall={mrecall:.4f}  mF1={mf1:.4f}")
    logger.info(f"  mSpecificity={mspecificity:.4f}  FWIoU={fwiou:.4f}")
    logger.info(f"\n{'='*80}")
    header = f"{'Class':>16s} {'IoU':>8s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Spec':>8s} {'GT#':>10s} {'Pred#':>10s}"
    logger.info(header)
    logger.info("-" * len(header))
    for c in range(num_classes):
        logger.info(
            f"{names[c]:>16s} {iou_per_class[c]:>8.4f} {acc_per_class[c]:>8.4f} "
            f"{precision_per_class[c]:>8.4f} {recall_per_class[c]:>8.4f} "
            f"{f1_per_class[c]:>8.4f} {specificity_per_class[c]:>8.4f} "
            f"{int(gt_count[c]):>10d} {int(pred_count[c]):>10d}"
        )
    logger.info(f"{'Mean':>16s} {miou:>8.4f} {macc:>8.4f} {mprecision:>8.4f} {mrecall:>8.4f} {mf1:>8.4f} {mspecificity:>8.4f}")

    logger.info(f"\nConfusion Matrix (row=GT, col=Pred):")
    gt_label = "GT\\Pred"
    header = f"{gt_label:>12s}" + "".join(f"{names[i][:8]:>9s}" for i in range(num_classes))
    logger.info(header)
    for c in range(num_classes):
        row = f"{names[c]:>12s}" + "".join(f"{total_confusion[c,i]:>9d}" for i in range(num_classes))
        logger.info(row)

    result_path = os.path.join(cfg.save_path, "eval_val_result.json")
    per_class_metrics = {}
    for c in range(num_classes):
        per_class_metrics[names[c]] = {
            "IoU": round(float(iou_per_class[c]), 4),
            "Accuracy": round(float(acc_per_class[c]), 4),
            "Precision": round(float(precision_per_class[c]), 4),
            "Recall": round(float(recall_per_class[c]), 4),
            "F1": round(float(f1_per_class[c]), 4),
            "Specificity": round(float(specificity_per_class[c]), 4),
            "GT_count": int(gt_count[c]),
            "Pred_count": int(pred_count[c]),
        }
    with open(result_path, "w") as f:
        json.dump({
            "mIoU": round(float(miou), 4),
            "mAcc": round(float(macc), 4),
            "allAcc": round(float(allacc), 4),
            "mPrecision": round(float(mprecision), 4),
            "mRecall": round(float(mrecall), 4),
            "mF1": round(float(mf1), 4),
            "mSpecificity": round(float(mspecificity), 4),
            "FWIoU": round(float(fwiou), 4),
            "per_class": per_class_metrics,
            "per_sample": per_sample,
            "confusion_matrix": total_confusion.tolist(),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {result_path}")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(main_worker, num_gpus_per_machine=args.num_gpus,
           num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, cfg=(cfg,))
