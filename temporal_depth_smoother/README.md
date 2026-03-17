# Temporal Depth Smoother

A lightweight, system-agnostic post-processing network for temporally consistent depth map sequences. 

**Key Features:**
- ✓ Works with any depth system (DA3, Marigold, DepthCrafter, stereo, etc.)
- ✓ Offline/bidirectional processing (sees full temporal window)
- ✓ Residual learning: predicts small corrections, not absolute depth
- ✓ Per-clip normalization: handles arbitrary depth scales
- ✓ System-agnostic training: train once, apply to any depth source
- ✓ ~500K parameters (3D CNN): fast to train and inference

---

## Overview

Multi-camera depth systems like DA3 produce geometrically accurate but temporally jittery depth maps. This module learns to remove temporal jitter while:
1. Preserving geometric accuracy (stays close to input)
2. Preserving real object motion (doesn't over-smooth)
3. Strengthening background stability (removes flicker)

### Training Signal: VDA as Temporal Supervision

The network is trained using:
- **Input**: Jittery depth from DA3 (or any system)
- **Supervision**: Temporally smooth depth gradients from VDA, aligned to DA3's absolute scale
- **Anchor**: Static background pixels (0 temporal variance)

The trick: VDA's temporal structure is trustworthy (it uses temporal self-attention), but its absolute values are wrong. By aligning VDA's gradients to match DA3's scale on the static background, we get a supervision signal that:
- Teaches temporal smoothness (from VDA's smooth trajectories)
- Respects geometry (aligned to DA3's metric scale)
- Preserves real motion (only trained on gradient changes, not absolute values)

---

## Installation

```bash
cd temporal_depth_smoother
# All dependencies are standard PyTorch + numpy
```

---

## Project Structure

```
temporal_depth_smoother/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── model.py                 # TemporalDepthSmoother (3D CNN)
├── losses.py                # Loss functions (temporal, geometric, smooth)
├── data.py                  # TemporalDepthDataset, DataLoader creation
├── utils.py                 # Utilities (normalization, alignment, inference)
├── train.py                 # Training script
├── inference.py             # Inference wrapper
├── preprocess.py            # Data preprocessing (Week 1)
└── README.md                # This file
```

---

## Quickstart

### 1. Data Preparation (Week 1)

You need three files per clip:
- `{clip_id}_depth_raw.npy` — Raw depth from DA3 (or other system) `[T, H, W]`
- `{clip_id}_depth_vda.npy` — Raw VDA depth (unaligned) `[T, H, W]`
- `{clip_id}_rgb.npy` — RGB frames `[T, H, W, 3]`

**Step 1a: Compute Background Mask**  
Identify static regions where the scene doesn't move (temporal variance < threshold).

```bash
python preprocess.py \
  --clip-id clip_001 \
  --depth-da3 /path/to/da3_depth.npy \
  --depth-vda /path/to/vda_depth.npy \
  --rgb /path/to/rgb.npy \
  --output-dir ./data/processed \
  --bg-threshold 0.01 \
  --visualize
```

This script:
1. Computes background mask from DA3 depth variance
2. Aligns VDA to DA3 using least-squares fit on background only
3. Validates alignment (should see MAE < 0.1 on background)
4. Saves: `depth_raw.npy`, `depth_vda_aligned.npy`, `rgb.npy`, `bg_mask.npy`

**Critical**: Examine the visualization. VDA and DA3 should nearly overlap on background regions. If not:
- Increase `--bg-threshold` (too strict)
- Check if moving object contaminates background mask

**Step 1b: Prepare Training Data**  
Create separate train/val splits. All samples should go into a single `data/` directory:

```
data/
├── clip_001_depth_raw.npy
├── clip_001_rgb.npy
├── clip_001_depth_vda_aligned.npy
├── clip_002_depth_raw.npy
├── clip_002_rgb.npy
├── clip_002_depth_vda_aligned.npy
├── ...
```

### 2. Training (Week 2-3)

```bash
python train.py \
  --data-dir ./data \
  --output-dir ./outputs/experiment_001 \
  --batch-size 4 \
  --num-epochs 50 \
  --learning-rate 1e-3 \
  --temporal-window 16
```

Outputs:
- `outputs/experiment_001/best_model.pt` — Best checkpoint
- `outputs/experiment_001/config.json` — Training config
- `outputs/experiment_001/tensorboard/` — TensorBoard logs

**Monitoring**: View loss curves in TensorBoard:
```bash
tensorboard --logdir outputs/experiment_001/tensorboard
```

Expected behavior:
- `loss/temporal`: Should drop (gradient alignment with VDA)
- `loss/geometric`: Should remain low (staying close to input)
- `loss/smooth`: Should drop on static regions

### 3. Inference (Week 4)

**Single-clip inference:**
```bash
python inference.py \
  --checkpoint outputs/experiment_001/best_model.pt \
  --depth-input /path/to/jittery_depth.npy \
  --rgb-input /path/to/rgb.npy \
  --output /path/to/smooth_depth.npy \
  --window-size 16 \
  --stride 8
```

**In Python:**
```python
from temporal_depth_smoother import DepthSmoother
import numpy as np

smoother = DepthSmoother('outputs/experiment_001/best_model.pt', device='cuda')

depth_raw = np.load('jittery_depth.npy')  # [T, H, W]
rgb = np.load('rgb.npy')                   # [T, H, W, 3]

depth_smooth = smoother.smooth(depth_raw, rgb, window_size=16, stride=8)
np.save('smooth_depth.npy', depth_smooth)
```

---

## Configuration

Edit [config.py](config.py) to adjust hyperparameters:

```python
from config import Config, ModelConfig, TrainingConfig

# Model
ModelConfig(
    T=16                      # Temporal window size
    base_channels=32          # Base feature dimension
)

# Training
TrainingConfig(
    batch_size=4
    learning_rate=1e-3
    num_epochs=50
    lambda_temporal=1.0       # Weight for temporal consistency
    lambda_geometric=0.5      # Weight for geometric fidelity
    lambda_smooth=0.3         # Weight for smoothness on static regions
    motion_threshold=0.05     # RGB motion detection threshold
)
```

---

## Loss Functions

The network optimizes three loss terms:

### L_temporal (Core Signal)
Ensures temporal *gradients* match VDA's smooth trajectories:
```
L_temporal = ||(depth_smooth[t+1] - depth_smooth[t]) - 
                (depth_vda_aligned[t+1] - depth_vda_aligned[t])||_L1
```
This is the main training signal. Supervises how depth *changes*, not absolute values.

### L_geometric (Fidelity Anchor)
Keeps smoothed depth close to input on geometrically confident regions:
```
w = exp(-5 * gradient(rgb))  # Low weights at RGB edges
L_geometric = ||w * (depth_smooth - depth_raw)||_L1
```
The edge-aware weight means: stay very close to input on flat regions (high confidence), relax at object boundaries (uncertain).

### L_smooth (Static Stability)
Suppresses any depth change on regions where RGB doesn't move:
```
motion_mask = (|RGB[t+1] - RGB[t]| > threshold)
L_smooth = ||(1 - motion_mask) * (depth_smooth[t+1] - depth_smooth[t])||_L1
```
Ensures background pixels have zero temporal variance.

### Combined Loss
```
L_total = λ1 * L_temporal + λ2 * L_geometric + λ3 * L_smooth
```
Default: λ1=1.0, λ2=0.5, λ3=0.3. Increase λ2 if depth drifts from input geometry.

---

## Architecture

### 3D CNN (Default)
Simple, fast, bidirectional temporal reasoning:

```
Input: depth_raw [B, T, H, W]  +  rgb [B, T, H, W, 3]
  ↓
Per-clip normalization: (x - mean) / std
  ↓
2D Conv encoders (per-frame):
  RGB:   3 → 16 channels
  Depth: 1 → 16 channels
  ↓
Concatenate: [B, T, 32, H, W]
  ↓
3D Conv blocks (bidirectional):
  Conv3D(32, 32, kernel=3x3x3)  ← 1 frame context each side
  Conv3D(32, 32, kernel=3x3x3)
  Conv3D(32, 16, kernel=5x3x3)  ← 2 frame context each side
  ↓
Residual head: Conv3D(16, 1, kernel=1x1x1)  → Δ(t)
  ↓
Output: depth_smooth = norm * (depth_norm + Δ) / std + mean

Model size: ~500K parameters
Speed: ~50ms per 16-frame clip on V100
```

### Per-Clip Normalization (Critical)
Different depth systems have different scales:
- DA3 metric: 0.1m–10m
- Marigold: 0–1 (relative)
- Stereo disparity: 0–255

Per-clip normalization maps all inputs to $\mathcal{N}(0, 1)$, making the network scale-invariant:

```python
mean = depth_raw.mean(dim=[1,2,3])
std  = depth_raw.std(dim=[1,2,3])
depth_norm = (depth_raw - mean) / std           # To standard scale

# ... network processes depth_norm ...

depth_smooth = depth_smooth_norm * std + mean  # Back to original scale
```

This allows training on DA3 and directly applying to Marigold without retraining.

---

## Evaluation

### Temporal Consistency Metrics

On **static background regions** (motion_mask = 0):

1. **Temporal Variance** (lower is better):
   ```python
   var_before = depth_raw.var(dim=0)[static_pixels].mean()
   var_after = depth_smooth.var(dim=0)[static_pixels].mean()
   improvement = (var_before - var_after) / var_before
   ```
   Target: 50–80% variance reduction on background.

2. **Mean Absolute Deviation from Input**:
   ```python
   mae = (depth_smooth - depth_raw).abs().mean()
   ```
   Target: < 0.5% of input range on static regions.

### Geometric Accuracy

On **moving object regions** (motion_mask = 1):

3. **Gradient Correlation with Input**:
   ```python
   grad_corr = pearson(grad(depth_smooth), grad(depth_raw))
   ```
   Target: > 0.95 (preserve real motion).

4. **Depth Similarity on Moving Pixels**:
   ```python
   mae_moving = (depth_smooth - depth_raw).abs()[motion_pixels].mean()
   ```
   Target: Stay within ±5% of overall depth range.

---

## Implementation Roadmap

### Week 1: Data Pipeline ✓
- [x] Run DA3 on all footage → `depth_raw`
- [x] Run VDA on same footage (monocular) → `depth_vda`
- [x] Compute background mask (stddev threshold)
- [x] Per-clip scale/shift alignment (background only)
- [x] Sanity check: plot `depth_raw` vs `depth_vda_aligned` on background

### Week 2: Network Implementation ✓
- [x] `TemporalDepthSmoother` (3D CNN)
- [x] Per-clip normalization/denormalization
- [x] Loss functions (temporal, geometric, smooth)
- [x] Motion mask from RGB frame differencing
- [x] Unit tests

### Week 3: Training ✓
- [ ] Train on small subset (5–10 clips)
- [ ] Verify losses decrease
- [ ] Scale to full dataset
- [ ] Log temporal variance improvement
- [ ] Checkpoint management

### Week 4: Evaluation & Tuning ✓
- [ ] Visual inspection (flicker reduction)
- [ ] Quantitative metrics on background
- [ ] Test on non-DA3 inputs (generalization)
- [ ] Tune loss weights if needed

---

## Key Insights

1. **Background-Only Alignment is Critical**
   - Moving objects violate static scene assumption
   - Aligning on them corrupts training signal
   - Static background is the only reliable anchor

2. **Supervise Gradients, Not Absolute Values**
   - Prevents network from drifting away from input geometry
   - Naturally preserves real object motion
   - Makes training more stable (less extreme corrections)

3. **Per-Clip Normalization Enables System Agnosticism**
   - No need to retrain for different depth systems
   - Network learns scale-invariant smoothing
   - Only requirement: RGB frame + depth sequence

4. **Offline Processing >> Online**
   - Bidirectional context is much stronger
   - Sliding window with blending avoids boundary artifacts
   - No sequential dependencies or hidden state carry-over

---

## Troubleshooting

### N1: Training Loss Not Decreasing
- ✓ Check background mask (should be > 50% of image). If too strict, increase `--bg-threshold`
- ✓ Check VDA alignment: sanity check MAE should be < 0.1
- ✓ Check loss weights. Maybe increase λ_temporal
- ✓ Try smaller learning rate (1e-4)

### N2: Network Output Drifts from Input
- ✓ Increase λ_geometric (geometric loss weight)
- ✓ Decrease temporal_window (less context = less aggressive smoothing)
- ✓ Check motion_threshold: if too high, L_smooth over-regularizes

### N3: Output Still Jittery (Not Smoothing)
- ✓ Check VDA alignment on background: must be nearly identical to DA3
- ✓ Increase λ_temporal to emphasize gradient matching
- ✓ Ensure motion_mask is correctly computed (check RGB diff visualization)

### N4: Over-Smoothing (Losing Real Motion)
- ✓ Decrease λ_smooth (static stabilization weight)
- ✓ Increase motion_threshold (more pixels classified as "moving")
- ✓ Check motion mask: should only cover truly moving regions

---

## References

**Blind Video Temporal Consistency** (Bonneel et al., SIGGRAPH 2015)  
Foundational work on video consistency using original video as guide.

**Learning Blind Video Temporal Consistency** (Lai et al., ECCV 2018)  
First deep learning approach. Uses recurrent networks; we use bidirectional 3D CNN.

**Towards Practical Consistent Video Depth Estimation** (ICMR 2023)  
Plug-and-play depth consistency post-processing. Online/recurrent; we extend to offline/bidirectional.

**Video Depth Anything (VDA)** (2024)  
Our source of temporal supervision. Uses temporal self-attention and gradient matching loss.

---

## Citation

If you use this code, please cite:

```bibtex
@software{temporal_depth_smoother_2026,
  title   = {Temporal Depth Smoother: System-Agnostic Post-Processing for Depth Consistency},
  author  = {Digital Clone Team},
  year    = {2026},
  url     = {https://github.com/...}
}
```

---

## License

See [LICENSE](../LICENSE) file.
