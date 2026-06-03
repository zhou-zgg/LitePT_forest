import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 10

LOG = "exp/forest/semseg-litept-small-v1m1_test5/train.log"

epochs, train_loss, val_loss, val_miou, val_macc, val_allacc = [], [], [], [], [], []
class_iou = {i: [] for i in range(7)}
class_names = ["terrain", "foliage", "CWD", "trunk", "snag", "non-tree", "branch"]

with open(LOG) as f:
    content = f.read()

# parse train result lines
for m in re.finditer(r'Train result: loss: ([\d.]+)', content):
    train_loss.append(float(m.group(1)))

# parse val result
for m in re.finditer(r'Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)\.', content):
    val_miou.append(float(m.group(1)))
    val_macc.append(float(m.group(2)))
    val_allacc.append(float(m.group(3)))

# parse class IoU
for i in range(7):
    for m in re.finditer(rf'Class_{i}-{class_names[i]} Result: iou/accuracy ([\d.]+)/([\d.]+)', content):
        class_iou[i].append((float(m.group(1)), float(m.group(2))))

epochs = list(range(1, len(val_miou) + 1))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Train loss
ax = axes[0, 0]
ax.plot(range(1, len(train_loss)+1), train_loss, 'b-', linewidth=1)
ax.set_title("Train Loss")
ax.set_xlabel("Iteration (epoch)")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)

# 2. Val metrics
ax = axes[0, 1]
ax.plot(epochs, val_miou, 'r-', label="mIoU", linewidth=1.5)
ax.plot(epochs, val_macc, 'g--', label="mAcc", linewidth=1)
ax.plot(epochs, val_allacc, 'b:', label="allAcc", linewidth=1)
ax.set_title("Validation Metrics")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Per-class IoU
ax = axes[1, 0]
for i in range(7):
    if class_iou[i]:
        ious = [x[0] for x in class_iou[i]]
        ax.plot(epochs[:len(ious)], ious, label=f"{i}-{class_names[i]}", linewidth=1.5)
ax.set_title("Per-Class IoU")
ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4. mIoU bar chart (last epoch)
ax = axes[1, 1]
last_ious = [class_iou[i][-1][0] if class_iou[i] else 0.0 for i in range(7)]
colors = ['#2ecc71' if iou > 0.1 else '#e74c3c' for iou in last_ious]
bars = ax.bar(class_names, last_ious, color=colors, alpha=0.8)
ax.set_title("Per-Class IoU (Last Epoch)")
ax.set_ylabel("IoU")
ax.set_ylim(0, 1)
for bar, iou in zip(bars, last_ious):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{iou:.2f}", ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=30)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("exp/forest/semseg-litept-small-v1m1_test5/training_curves.png", dpi=150)
print("Saved: exp/forest/semseg-litept-small-v1m1_test5/training_curves.png")

# Print summary table
print("\n=== Per-Class IoU Summary ===")
print(f"{'Class':<12} {'Last IoU':>10} {'Best IoU':>10}")
print("-" * 34)
for i in range(7):
    if class_iou[i]:
        last = class_iou[i][-1][0]
        best = max(x[0] for x in class_iou[i])
        print(f"{class_names[i]:<12} {last:>10.4f} {best:>10.4f}")
    else:
        print(f"{class_names[i]:<12} {'N/A':>10} {'N/A':>10}")

print(f"\nBest mIoU: {max(val_miou):.4f} (epoch {val_miou.index(max(val_miou))+1})")
print(f"Final mIoU: {val_miou[-1]:.4f}")
