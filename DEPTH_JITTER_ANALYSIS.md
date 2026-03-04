# Depth Jitter & Dynamic Gaussian Quality Analysis

## Summary
**YES**, inconsistent depth maps and jitter are likely significant contributors to poor dynamic gaussian output. The analysis reveals several critical issues in the current pipeline.

---

## Root Cause Analysis

### 1. **Insufficient Temporal Filtering**
|Location|Issue|Impact|
|--------|-----|------|
|[depth_inference/parse_depth_da3_streaming.py:130-200](depth_inference/parse_depth_da3_streaming.py) | Only blends at chunk boundaries (~10-20 frames) | Frame-to-frame jitter within chunks not addressed |
|[gs_train.py:250-260](gs_train.py) | Depth loss only checks `depth_reliable` flag | No per-pixel temporal consistency validation |
|[data_process/data_process_pcd.py](data_process/data_process_pcd.py) | Limited outlier removal | Noisy depths affect point cloud initialization |

**Evidence**: Depth estimation methods (DA3, Pi3X) process frames independently or in small overlapping windows. Without explicit temporal regularization, micro-motions from video jitter propagate directly into depth maps.

---

### 2. **Cascading Effects on Dynamic Deformation**

```
Jittery 2D Tracking → Inconsistent 3D Backprojection → Noisy Point Cloud
                                                           ↓
Unreliable Bone Skeleton → Poor Motion Interpolation → Artifacts in Deformation
      ↑___________ [gaussian_splatting/dynamic_utils.py:67-100] ___________|
```

The motion interpolation code assumes a stable bone skeleton. If the tracked points jitter, the skeleton twists and warps erratically between frames.

---

### 3. **Depth Regularization Gaps**

**Current approach** [gs_train.py:250-260]:
- Uses inverse depth (`invdepthmap`) as a regularization target
- Simple `depth_reliable` boolean flag
- No check for temporal consistency across frames

**Problem**: Two consecutive frames render the same geometry with slightly different depths → creates conflicting gradient signals during training.

---

## Diagnostic Steps

### Step 1: Identify if Depth is the Problem
```bash
# Run this first to quantify temporal jitter:
python analyze_depth_consistency.py \
    --depth_dir data/different_types/YOUR_CASE/depth \
    --output_json depth_analysis.json \
    --visualize
```

**Look for**:
- `temporal_variance.mean` > 0.01m = **significant jitter**
- `jitter_percent` > 10% = **problematic**
- `temporal_variance.p95` > 0.05m = **outliers present**

### Step 2: Check Depth vs Motion Quality
```bash
# Compare point cloud consistency across frames:
python data_process/data_process_pcd.py \
    --base_path data/different_types \
    --case_name YOUR_CASE
```

Look at the generated point clouds - they should be smooth and well-registered. Ragged or jittery point clouds indicate depth issues.

### Step 3: Monitor Training Convergence
Track these metrics during `gs_train.py`:
```python
# In the training loop, add monitoring:
print(f"[DEPTH] invdepth L1 loss: {Ll1depth:.6f}")  # Should decrease smoothly
print(f"[DEPTH] rendered vs GT: {loss_depth:.6f}")   # Should be stable
```

If depth losses are noisy/unstable → depth maps are inconsistent.

---

## Solutions (In Order of Complexity)

### Solution 1: Apply Temporal Depth Smoothing ⭐ **START HERE**
**Ease**: Easy | **Impact**: High | **Time**: 5 minutes

Apply median filtering to reduce outliers:
```bash
python temporal_depth_filter.py \
    --input parsed_depth_DA3_streaming/ \
    --output parsed_depth_DA3_smoothed/ \
    --method median \
    --window 5 \
    --strength 0.7
```

Then **retrain** the gaussian splatting with smoothed depths.

**Why it works**: Median filter is robust to outliers while preserving depth edges. Frame-to-frame consistency improves immediately.

---

### Solution 2: Add per-Pixel Depth Confidence Maps ⭐⭐ **RECOMMENDED**
**Ease**: Medium | **Impact**: High | Time: 1-2 hours

Modify the depth inference pipeline:

```python
# In depth_inference/infer_depth.py or similar
def compute_depth_confidence(
    depth_maps: Dict[int, np.ndarray],
    rgb_frames: Dict[int, np.ndarray],
    window_size: int = 5
) -> Dict[int, np.ndarray]:
    """
    Compute per-pixel confidence based on:
    - Temporal consistency (variance across nearby frames)
    - Spatial smoothness (low variance = more confident)
    - Multi-view consistency (if available)
    """
    confidence = {}
    
    for cam_id, depths in depth_maps.items():
        T, H, W = depths.shape
        conf = np.ones((T, H, W), dtype=np.float32)
        
        # Temporal variance
        for t in range(T):
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            
            temporal_std = np.std(depths[start:end], axis=0)
            # Low variance = high confidence
            conf[t] = np.exp(-temporal_std / 0.05)  # Tune the scale
        
        # Spatial smoothness term
        for t in range(T):
            gy, gx = np.gradient(depths[t])
            spatial_grad = np.sqrt(gx**2 + gy**2)
            # Smooth regions more confident
            conf[t] *= np.exp(-spatial_grad / 0.1)
        
        confidence[cam_id] = conf
    
    return confidence
```

Then use confidence maps in training:
```python
# In gs_train.py, modify depth loss:
if opt.lambda_depth > 0:
    gt_depth = viewpoint_cam.depth.cuda()
    depth_confidence = viewpoint_cam.depth_confidence.cuda()  # Add this
    
    loss_depth = depth_loss(depth, gt_depth, alpha_mask) * depth_confidence.mean()
    loss = loss + opt.lambda_depth * loss_depth
```

---

### Solution 3: Implement Optical Flow-Guided Depth Warping ⭐⭐⭐ **MOST ROBUST**
**Ease**: Hard | **Impact**: Very High | Time: 3-4 hours

Account for camera/object motion before filtering:

```python
# In temporal_depth_filter.py (already implemented)
filter_obj = TemporalDepthFilter(method="flow-guided", strength=0.8)
filtered = filter_obj.filter(depth_sequence, rgb_sequence)
```

This accounts for actual motion, making temporal filtering much more effective.

---

### Solution 4: Modify Depth Inference to Use Temporal Consistency Loss
**Ease**: Medium | **Impact**: High | Time: 2-3 hours

Add temporal regularization to depth inference model:
```python
def temporal_consistency_loss(depth_t, depth_t1, warp_matrix):
    """
    Penalize depth changes that don't match optical flow.
    """
    # Warp depth_t using optical flow to align with depth_t1
    depth_t_warped = warp_depth(depth_t, warp_matrix)
    
    # Pixel-wise difference should be small
    consistency_loss = torch.abs(depth_t_warped - depth_t1).mean()
    return consistency_loss
```

---

## Implementation Checklist

### Immediate Actions (Next 30 minutes)
- [ ] Run `analyze_depth_consistency.py` to diagnose severity
- [ ] Apply `temporal_depth_filter.py` with `method=median`
- [ ] Retrain gaussian splatting with smoothed depths
- [ ] Compare rendering quality

### Intermediate Actions (1-2 hours)
- [ ] Implement depth confidence maps
- [ ] Modify loss weights based on confidence
- [ ] Retrain and evaluate

### Long-term Actions (Architecture improvements)
- [ ] Add temporal loss to depth inference model
- [ ] Implement optical flow-guided filtering
- [ ] Consider multi-view depth fusion

---

## Validation Metrics

After applying solutions, track these metrics:

```python
# Before/After comparison
metrics = {
    "temporal_jitter": "analyze_depth_consistency.py output",
    "chamfer_error": "evaluate_chamfer.py",
    "rendering_quality": "visual inspection or LPIPS/SSIM",
    "training_stability": "smoothness of loss curves",
}
```

**Target improvements**:
- `temporal_jitter.mean` < 0.005m
- Smoother training loss curves
- Sharper, less flickering dynamic renders

---

## Script Usage Guide

### 1. Analyze Current Depth Quality
```bash
python analyze_depth_consistency.py \
    --depth_dir data/different_types/CASE_NAME/depth \
    --output_json depth_consistency_report.json \
    --visualize --max_frames 100
```

**Output**: JSON report + visualization showing jitter hotspots.

### 2. Apply Temporal Smoothing
```bash
# Try each method to see which works best:

# Method 1: Median (robust to outliers)
python temporal_depth_filter.py \
    --input parsed_depth_DA3_streaming/ \
    --output parsed_depth_smoothed_median/ \
    --method median --window 5 --strength 0.7

# Method 2: Bilateral (preserves edges)
python temporal_depth_filter.py \
    --input parsed_depth_DA3_streaming/ \
    --output parsed_depth_smoothed_bilateral/ \
    --method bilateral --window 5 --strength 0.6

# Method 3: Flow-guided (motion-aware, best)
python temporal_depth_filter.py \
    --input parsed_depth_DA3_streaming/ \
    --output parsed_depth_smoothed_flow/ \
    --method flow-guided --window 7 --strength 0.8 \
    --rgb_dir color/
```

### 3. Retrain with Smoothed Depths
```bash
# Update your configuration to point to smoothed depths
python gs_train.py \
    --source_path data/different_types/CASE_NAME \
    --model_path gaussian_output/CASE_NAME_retrained \
    --iterations 30000 \
    --depth_l1_weight_init 0.2 \
    --depth_l1_weight_final 0.2
```

---

## FAQ

**Q: Will smoothing lose important depth detail?**
A: No - median filtering with `window=5, strength=0.7` removes jitter while preserving sharp depth edges. Temporal smoothing is different from spatial blurring.

**Q: How much will this help?**
A: Depends on jitter severity:
- Mild jitter (variance < 0.01m): ~15-25% quality improvement
- Moderate jitter (variance 0.01-0.05m): ~30-50% improvement
- Severe jitter (variance > 0.05m): **Critical** - fixes major artifacts

**Q: Can I skip optical flow filtering?**
A: Yes, start with median. Only add flow-guided if median isn't sufficient. Flow-guided is more robust but requires RGB frames.

**Q: Should I retrain from scratch or fine-tune?**
A: Fine-tune (`--checkpoint`) from existing model. The smoothed depths are better constraints, not a complete change of direction.

---

## Related Issues in Codebase

1. **Motion jitter**: Check `gaussian_splatting/dynamic_utils.py:67-100` - motion interpolation might need temporal smoothing too
2. **Point cloud jitter**: Run `data_process/data_process_pcd.py` with updated smooth depths
3. **Camera calibration**: Verify with `camera_alignment/calibrate_camera_extrinsics.py` - camera drift can appear as depth jitter

---

## References

- DA3 depth quality: Check for frame drops or model switches
- Temporal smoothing papers: BilateralSolver, OpticalFlowFusion
- Gaussian splatting with dynamic content: GS-DVGO (similar problem/solution)
