# Architecture Decision Records (ADR)

## ADR-001: GridSample before CylinderCrop

**Status**: Accepted

**Date**: 2026-07-06

**Context**: Point cloud transforms pipeline order determines how points flow through voxelization and cropping. Two options: (A) GridSample → Crop (original), (B) Crop → GridSample.

**Decision**: Keep GridSample before CylinderCrop (Option A).

**Rationale**: 
- GridSample first ensures consistent voxel representation across all crops
- Swapping order (Crop→GridSample) caused BatchNorm failures due to insufficient points per batch
- The post-voxelization class distribution is more predictable for oversampling
- Trade-off: NTC points collapse 99% at 0.02m grid (dense scanning on thin poles). Solved by ADR-002.

---

## ADR-002: Oversampling via WeightedRandomSampler

**Status**: Accepted

**Date**: 2026-07-08

**Context**: NTC (utility poles, class 6) was never predicted (IoU=0.0) across 19 epochs of V21. Root cause: NTC in only 14/91 training files (15% of batches) and post-voxelization only 2% of points per batch.

**Decision**: Implement file-level oversampling using `WeightedRandomSampler` with adaptive weights computed from per-file class histograms.

**Rationale**:
- File-level rarity (`n_files_sans_class_c / n_files_total`) directly addresses the batch frequency problem
- Adaptive: weights auto-recompute when new data is added (JSON-cached histograms)
- NTC batch frequency: 15% → 42% → first prediction at epoch 1
- Alternative rejected: class-weighted CE (V25/V26) hurt SNG (-0.21 IoU)

**Implementation**: `ForestDataset._compute_sample_weights()` computes per-file weights from cached JSON class histograms. `engines/train.py` checks for `dataset.sample_weights` and creates `WeightedRandomSampler`.

---

## ADR-003: Grid size 0.02m → 0.025m

**Status**: Accepted

**Date**: 2026-07-09

**Context**: Terrain IoU oscillated 0.53-0.80 with grid_size=0.02m and 450K crop. Model lacked spatial context for terrain (a large, smooth surface class).

**Decision**: Increase grid_size from 0.02m to 0.025m.

**Rationale**:
- +56% spatial coverage for same 450K point budget ((0.025/0.02)² = 1.56x area)
- Terrain IoU: 0.54 → 0.80 (V22→V23) and stabilized (no more oscillation)
- Zero memory cost (fewer voxels per crop)
- Trade-off: NTC voxels reduced 5-7 → 4-6 per pole cross-section; branch 1-2.5 → 0.8-2 voxels per cross-section
- Branch did NOT degrade (0.612→0.639) thanks to ADR-004

---

## ADR-004: Encoder patch_size 1024 → 2048 for deep stages

**Status**: Accepted

**Date**: 2026-07-09

**Context**: V17 (Small model) used enc_patch_size=(1024,1024,1024,2048,2048) and achieved terrain=0.8525. V22 (Base model) used enc_patch_size=(1024,1024,1024,1024,1024) and terrain=0.5356.

**Decision**: Restore V17's patch_size configuration for Base model: deep stages 3-4 use patch_size=2048.

**Rationale**:
- 2x larger attention window in deep layers (stages 3-4: 16K→33K input points per window)
- FlashAttention enabled (PyTorch 2.4.1): memory scales O(n) not O(n²) — negligible memory cost
- Protects branch from coarser grid (ADR-003): branch 0.612→0.639 despite 0.025m grid
- Combined with ADR-003: mIoU improved 0.6007→0.6329

---

## ADR-005: Weighted CrossEntropyLoss rejected

**Status**: Rejected

**Date**: 2026-07-11

**Context**: V25/V26 attempted to add class-weighted CrossEntropyLoss (weight=[1,1,1,2,2,5,10] and adaptive inverse_sqrt) on top of oversampling.

**Decision**: Reject weighted CE. Oversampling alone is sufficient.

**Rationale**:
- V26 best mIoU = 0.6320 vs V24 = 0.6635 (-0.031)
- NTC improved 0.724→0.777 (+0.05) but SNG collapsed 0.768→0.560 (-0.21) and trunk 0.866→0.779 (-0.09)
- Weighted CE over-amplifies rare class gradients, disrupting LovaszLoss/DiceLoss class-balanced optimization
- Net negative effect on overall performance

---

## ADR-006: Server as source of truth

**Status**: Accepted

**Date**: 2026-07-08

**Context**: Code development on local machine, training on remote server (RTX 3090, 49GB). Need to avoid divergence.

**Decision**: Server repo is the canonical codebase. All commits happen on server. Local only for editing and syncing.

**Rationale**:
- Training configs, experiment outputs only exist on server
- Server has CUDA extensions built for GPU architecture
- Local code falls behind quickly; sync via scp, not git push
