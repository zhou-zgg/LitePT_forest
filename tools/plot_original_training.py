from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SAVE_DIR = "exp/forest/semseg-litept-small-v1m1"
EVENT_FILE = f"{SAVE_DIR}/events.out.tfevents.1777366182.caozhou-MS-7D31"

ea = event_accumulator.EventAccumulator(EVENT_FILE)
ea.Reload()

tags = ea.Tags()['scalars']
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 1. Main training curves ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

steps = [m.step for m in ea.Scalars('val/mIoU')]
values = [m.value for m in ea.Scalars('val/mIoU')]
best_idx = values.index(max(values))
best_epoch = steps[best_idx]

axes[0, 0].plot(steps, values, 'b-', linewidth=1.5)
axes[0, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('mIoU')
axes[0, 0].set_title(f'Validation mIoU (best={max(values):.4f} @ epoch {best_epoch})')
axes[0, 0].grid(True, alpha=0.3)

train_loss = [m.value for m in ea.Scalars('train/loss')]
val_loss = [m.value for m in ea.Scalars('val/loss')]
axes[0, 1].plot(steps, train_loss, 'b-', linewidth=1, label='Train', alpha=0.7)
axes[0, 1].plot(steps, val_loss, 'r-', linewidth=1, label='Val', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Train vs Val Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

macc = [m.value for m in ea.Scalars('val/mAcc')]
axes[1, 0].plot(steps, macc, 'g-', linewidth=1.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('mAcc')
axes[1, 0].set_title('Validation mAcc')
axes[1, 0].grid(True, alpha=0.3)

allacc = [m.value for m in ea.Scalars('val/allAcc')]
axes[1, 1].plot(steps, allacc, 'm-', linewidth=1.5)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('allAcc')
axes[1, 1].set_title('Validation allAcc')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Original Training from Scratch - Training Overview (100 Epochs)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_overview.png", dpi=150, bbox_inches='tight')
print(f"Saved: {SAVE_DIR}/training_overview.png")

# ========== 2. Per-class IoU ==========
fig, ax = plt.subplots(figsize=(14, 7))
class_names = ['terrain', 'foliage', 'CWD', 'trunk', 'snag', 'non-tree', 'branch']
colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

for i, name in enumerate(class_names):
    tag = f'val/cls_{i}-{name}_IoU'
    if tag in tags:
        vals = [m.value for m in ea.Scalars(tag)]
        ax.plot(steps, vals, color=colors[i], linewidth=1.5, label=f'{name} (final={vals[-1]:.3f})', alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('IoU', fontsize=12)
ax.set_title('Per-Class IoU During Training', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/per_class_iou.png", dpi=150, bbox_inches='tight')
print(f"Saved: {SAVE_DIR}/per_class_iou.png")

# ========== 3. LR schedule ==========
fig, ax = plt.subplots(figsize=(12, 5))
lr_steps = [m.step for m in ea.Scalars('params/lr')]
lr = [m.value for m in ea.Scalars('params/lr')]
ax.plot(lr_steps, lr, 'c-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/lr_schedule.png", dpi=150, bbox_inches='tight')
print(f"Saved: {SAVE_DIR}/lr_schedule.png")

# ========== 4. Final per-class bar chart ==========
fig, ax = plt.subplots(figsize=(12, 6))
final_class_iou = []
for i, name in enumerate(class_names):
    tag = f'val/cls_{i}-{name}_IoU'
    if tag in tags:
        vals = [m.value for m in ea.Scalars(tag)]
        final_class_iou.append(vals[-1])
    else:
        final_class_iou.append(0)

bars = ax.bar(class_names, final_class_iou, color=colors, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('IoU', fontsize=12)
ax.set_title(f'Final Per-Class IoU (Epoch {steps[-1]})', fontsize=14)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, final_class_iou):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_class_iou.png", dpi=150, bbox_inches='tight')
print(f"Saved: {SAVE_DIR}/final_class_iou.png")

# ========== 5. Train batch loss ==========
fig, ax = plt.subplots(figsize=(12, 5))
batch_loss = [m.value for m in ea.Scalars('train_batch/loss')]
ax.plot(range(len(batch_loss)), batch_loss, 'b-', linewidth=0.5, alpha=0.6)
ax.set_xlabel('Batch')
ax.set_ylabel('Loss')
ax.set_title('Training Batch Loss (every batch)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/train_batch_loss.png", dpi=150, bbox_inches='tight')
print(f"Saved: {SAVE_DIR}/train_batch_loss.png")

print(f"\nAll plots saved to {SAVE_DIR}/")
print(f"  training_overview.png")
print(f"  per_class_iou.png")
print(f"  lr_schedule.png")
print(f"  final_class_iou.png")
print(f"  train_batch_loss.png")
